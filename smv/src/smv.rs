use base::atom::Atom;
use base::module::Module;
use base::term::Term;
use base::wire::Wire;
use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;

use crate::dtype::DType;
use crate::itype::IType;

/// Return a twin of wires by shifting every index by `offset` (signed).
/// Returns Err on overflow/underflow.
/// TODO: this method of creating next variables should be removed in the future
pub fn twin(wires: &[Wire<DType>], offset: isize) -> Result<Vec<Wire<DType>>, &'static str> {
    let mut shifted: Vec<Wire<DType>> = Vec::with_capacity(wires.len());
    for wire in wires.iter() {
        let ni = (wire.id() as isize).checked_add(offset).ok_or("index overflow")?;
        if ni < 0 {
            return Err("index underflow");
        }
        shifted.push(Wire::new(ni as usize, *wire.dtype()));
    }
    Ok(shifted)
}

// Helper: classify read wires returned by expression parsing into latched
// vs ivar (wait) wires. `var_count` is the number of VAR declarations;
// indices < var_count are latched, indices >= var_count are IVARs.
fn classify_reads(
    read: &[Wire<DType>],
    var_count: usize,
    read_latched: &mut Vec<Wire<DType>>,
    wait_latched: &mut Vec<Wire<DType>>,
) {
    // Helper: append a single wire into target if not present
    fn add_unique(target: &mut Vec<Wire<DType>>, idx: usize, dtype: DType) {
        if !target.iter().any(|w| w.id() == idx) {
            target.push(Wire::new(idx, dtype));
        }
    }

    for wire in read.iter() {
        let ri = wire.id();
        let rdtype = wire.dtype();
        if ri < var_count {
            add_unique(read_latched, ri, *rdtype);
        } else {
            add_unique(wait_latched, ri, *rdtype);
        }
    }
}

// Small wrapper that returns (IType, read) by calling the main build_expr.
fn build_expr_no_write(
    expr_pair: Pair<Rule>,
    wires: &[(String, usize, DType)],
    offset: usize,
) -> (IType, Wire<DType>) {
    let (itype, _write, read) = build_expr(expr_pair, wires, offset);
    (itype, read)
}

// Lower an expression into a sequence of Terms. `final_write` when provided
// is the wire where the final result must be written (e.g., the next wire of
// a variable). `temp_index` is a mutable counter for allocating temporary
// wires starting at 2*n. `terms` is appended with the generated Terms.
#[allow(clippy::only_used_in_recursion)]
fn lower_expr_to_terms(
    expr_pair: Pair<Rule>,
    wires: &[(String, usize, DType)],
    var_count: usize,
    n: usize,
    final_write: Option<Wire<DType>>,
    temp_index: &mut usize,
    terms: &mut Vec<Term<DType, IType>>,
) -> (Wire<DType>, Vec<Wire<DType>>) {
    let rule = expr_pair.as_rule();
    // helper to ensure that when a caller requested `final_write`, the
    // returned wire is that final_write. If the lowered result wrote into
    // a temporary, we append an explicit Assign to move it into final_write.
    fn enforce(
        final_write: &Option<Wire<DType>>,
        terms: &mut Vec<Term<DType, IType>>,
        w: Wire<DType>,
        r: Vec<Wire<DType>>,
    ) -> (Wire<DType>, Vec<Wire<DType>>) {
        match final_write {
            Some(fw) if fw != &w => {
                terms.push(Term::function(IType::Assign, [fw.clone()], vec![w.clone()]).unwrap());
                (fw.clone(), r)
            }
            _ => (w, r),
        }
    }
    match rule {
        Rule::expr_cond => {
            // More deterministic lowering for conditional expressions so the
            // emitted operator Terms match the manual lowering ordering.
            let mut inner = expr_pair.into_inner();
            let cond_pair = inner.next().unwrap();
            // if there's no '?', just lower the condition expression
            if inner.peek().is_none() {
                return lower_expr_to_terms(
                    cond_pair,
                    wires,
                    var_count,
                    n,
                    final_write,
                    temp_index,
                    terms,
                );
            }
            inner.next(); // skip '?'
            let then_pair = inner.next().unwrap();
            inner.next(); // skip ':'
            let else_pair = inner.next().unwrap();

            // Lower condition structurally (no string heuristics). We keep
            // the returned read-union for classification and the actual
            // temporary wire produced for the term read list.
            let mut cond_read_union = vec![];
            let (cond_temp_wire, cond_temp_read) =
                lower_expr_to_terms(cond_pair, wires, var_count, n, None, temp_index, terms);
            cond_read_union.extend_from_slice(&cond_temp_read);
            let cond_temp = (cond_temp_wire, cond_read_union.clone());

            // cond_temp is a tuple (wire, read_union)
            let (cond_temp_wire, cond_temp_read) = cond_temp;

            // Lower the then branch. If it's arithmetic, lower constants and add.
            let (then_temp_wire, then_read) = match then_pair.as_rule() {
                Rule::expr_arith => {
                    // expect form: ident PLUS number
                    let mut ar = then_pair.into_inner();
                    let left = ar.next().unwrap();
                    let _op = ar.next().unwrap();
                    let right = ar.next().unwrap();
                    // lower right (likely a number)
                    let (right_w, right_read) =
                        lower_expr_to_terms(right, wires, var_count, n, None, temp_index, terms);
                    let (left_w, left_read) =
                        lower_expr_to_terms(left, wires, var_count, n, None, temp_index, terms);
                    // combine reads
                    let mut then_reads = vec![];
                    then_reads.extend_from_slice(&right_read);
                    then_reads.extend_from_slice(&left_read);
                    // ensure consts are available as temps (if right_w is a const term it has been emitted)
                    let add_w = {
                        let w = Wire::new(*temp_index, DType::Int);
                        *temp_index += 1;
                        w
                    };
                    let read = vec![left_w.clone(), right_w.clone()];
                    terms.push(Term::function(IType::Add, [add_w.clone()], read).unwrap());
                    // propagate original-variable reads upward
                    let mut then_read_union = vec![];
                    then_read_union.extend_from_slice(&then_reads);
                    (add_w, then_read_union)
                }
                _ => lower_expr_to_terms(then_pair, wires, var_count, n, None, temp_index, terms),
            };

            // Lower the else branch (likely a number)
            let (else_temp_wire, else_read) =
                lower_expr_to_terms(else_pair, wires, var_count, n, None, temp_index, terms);

            // final cond write: write into final_write (usually the next wire for the var)
            let write = match &final_write {
                Some(w) => w.clone(),
                None => {
                    let w = Wire::new(*temp_index, DType::Int);
                    *temp_index += 1;
                    w
                }
            };

            // The Term's read field must reference the actual wires (temps or consts)
            // produced for the condition, then-branch and else-branch so e.g. the
            // Cond reads the OR/temp bool and the computed then/else ints.
            let term_read_wires = vec![cond_temp_wire.clone(), then_temp_wire.clone(), else_temp_wire.clone()];
            terms.push(Term::function(
                IType::Cond,
                [write.clone()],
                term_read_wires.clone(),
            ).unwrap());

            // The returned read-union (for classification) remains the union of
            // original-variable reads from each subexpression.
            let mut orig_read_union = vec![];
            orig_read_union.extend_from_slice(&cond_temp_read);
            orig_read_union.extend_from_slice(&then_read);
            orig_read_union.extend_from_slice(&else_read);
            enforce(&final_write, terms, write, orig_read_union)
        }

        Rule::expr_or | Rule::expr_and => {
            let mut inner = expr_pair.into_inner();
            let mut operand_wires: Vec<Wire<DType>> = vec![];
            let mut operands_read_union = vec![];
            // first operand
            if let Some(first) = inner.next() {
                let (w, r) =
                    lower_expr_to_terms(first, wires, var_count, n, None, temp_index, terms);
                operand_wires.push(w);
                operands_read_union.extend_from_slice(&r);
            }
            while let Some(_) = inner.next() {
                if let Some(rhs) = inner.next() {
                    let (w, r) =
                        lower_expr_to_terms(rhs, wires, var_count, n, None, temp_index, terms);
                    operand_wires.push(w);
                    operands_read_union.extend_from_slice(&r);
                }
            }
            if operand_wires.len() == 1 {
                return (operand_wires[0].clone(), operands_read_union);
            }
            let write = match &final_write {
                Some(w) => w.clone(),
                None => {
                    let w = Wire::new(*temp_index, DType::Bool);
                    *temp_index += 1;
                    w
                }
            };
            let mut read = vec![];
            for ow in operand_wires.iter() {
                read.push(ow.clone());
            }
            let op_tag = if rule == Rule::expr_or {
                IType::Or
            } else {
                IType::And
            };
            terms.push(Term::function(op_tag, [write.clone()], read).unwrap());
            enforce(&final_write, terms, write, operands_read_union)
        }

        Rule::expr_cmp => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (mut left_w, left_read) =
                lower_expr_to_terms(first, wires, var_count, n, None, temp_index, terms);
            // produce a comparison term for each op
            let mut cmp_read_union = left_read.clone();
            while let Some(op) = inner.next() {
                let right_pair = inner.next().unwrap();
                let (right_w, right_read) =
                    lower_expr_to_terms(right_pair, wires, var_count, n, None, temp_index, terms);
                let write = Wire::new(*temp_index, DType::Bool);
                *temp_index += 1;
                let read = vec![left_w.clone(), right_w.clone()];
                let tag = match op.as_rule() {
                    Rule::LT => IType::Lt,
                    Rule::LE => IType::Le,
                    Rule::GT => IType::Gt,
                    Rule::GE => IType::Ge,
                    Rule::EQ => IType::Eq,
                    _ => IType::Lt,
                };
                terms.push(Term::function(tag, [write.clone()], read).unwrap());
                // accumulate original-variable reads
                cmp_read_union.extend_from_slice(&right_read);
                left_w = write;
            }
            enforce(&final_write, terms, left_w, cmp_read_union)
        }

        Rule::expr_arith => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (mut left_w, mut left_read) =
                lower_expr_to_terms(first, wires, var_count, n, None, temp_index, terms);
            while let Some(op) = inner.next() {
                let right_term = inner.next().unwrap();
                let (right_w, right_read) =
                    lower_expr_to_terms(right_term, wires, var_count, n, None, temp_index, terms);
                let write = Wire::new(*temp_index, DType::Int);
                *temp_index += 1;
                let read = vec![left_w.clone(), right_w.clone()];
                let tag = match op.as_rule() {
                    Rule::PLUS => IType::Add,
                    Rule::MINUS => IType::Sub,
                    _ => IType::Add,
                };
                terms.push(Term::function(tag, [write.clone()], read).unwrap());
                // accumulate original-variable reads
                left_read.extend_from_slice(&right_read);
                left_w = write;
            }
            enforce(&final_write, terms, left_w, left_read)
        }

        Rule::expr_term => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (mut left_w, mut left_read) =
                lower_expr_to_terms(first, wires, var_count, n, None, temp_index, terms);
            while let Some(op) = inner.next() {
                let right_primary = inner.next().unwrap();
                let (right_w, right_read) = lower_expr_to_terms(
                    right_primary,
                    wires,
                    var_count,
                    n,
                    None,
                    temp_index,
                    terms,
                );
                let write = Wire::new(*temp_index, DType::Int);
                *temp_index += 1;
                let read = vec![left_w.clone(), right_w.clone()];
                let tag = match op.as_rule() {
                    Rule::TIMES => IType::Mul,
                    Rule::DIVIDE => IType::Div,
                    _ => IType::Mul,
                };
                terms.push(Term::function(tag, [write.clone()], read).unwrap());
                left_read.extend_from_slice(&right_read);
                left_w = write;
            }
            (left_w, left_read)
        }

        Rule::expr_factor => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            if first.as_rule() == Rule::NOT {
                let inner_factor = inner.next().unwrap();
                let (sub_wire, sub_read) =
                    lower_expr_to_terms(inner_factor, wires, var_count, n, None, temp_index, terms);
                let write = match &final_write {
                    Some(w) => w.clone(),
                    None => {
                        Wire::new(*temp_index, DType::Bool);
                        *temp_index += 1;
                        Wire::new(*temp_index - 1, DType::Bool)
                    }
                };
                terms.push(Term::function(IType::Not, [write.clone()], vec![sub_wire]).unwrap());
                enforce(&final_write, terms, write, sub_read)
            } else {
                lower_expr_to_terms(first, wires, var_count, n, final_write, temp_index, terms)
            }
        }

        Rule::expr_assign => {
            // assignment inside expression: treat RHS lowering and return its write
            let mut inner = expr_pair.into_inner();
            let _target = inner.next().unwrap();
            let value_expr = inner.next().unwrap();
            lower_expr_to_terms(
                value_expr,
                wires,
                var_count,
                n,
                final_write,
                temp_index,
                terms,
            )
        }

        Rule::number => {
            let sval = expr_pair.as_str();
            let parsed = sval.parse::<i64>().expect("invalid numeric literal");
            let write = match &final_write {
                Some(w) => w.clone(),
                None => {
                    Wire::new(*temp_index, DType::Int);
                    *temp_index += 1;
                    Wire::new(*temp_index - 1, DType::Int)
                }
            };
            terms.push(Term::function::<Wire<DType>, Wire<DType>, _, _>(
                IType::ConstInt(parsed),
                [write.clone()],
                vec![],
            ).unwrap());
            enforce(&final_write, terms, write, vec![])
        }

        Rule::ident => {
            let name = expr_pair.as_str().to_string();
            if let Some((_, idx, dtype)) = wires.iter().find(|(n, _, _)| n == &name) {
                let wire = Wire::new(*idx, *dtype);
                enforce(
                    &final_write,
                    terms,
                    wire.clone(),
                    vec![wire],
                )
            } else {
                // Return a dummy wire with index 0 for error cases
                enforce(&final_write, terms, Wire::new(0, DType::Bool), vec![])
            }
        }

        Rule::TRUE => {
            let write = match &final_write {
                Some(w) => w.clone(),
                None => {
                    Wire::new(*temp_index, DType::Bool);
                    *temp_index += 1;
                    Wire::new(*temp_index - 1, DType::Bool)
                }
            };
            terms.push(Term::function::<Wire<DType>, Wire<DType>, _, _>(
                IType::ConstBool(true),
                [write.clone()],
                vec![],
            ).unwrap());
            enforce(&final_write, terms, write, vec![])
        }

        Rule::FALSE => {
            let write = match &final_write {
                Some(w) => w.clone(),
                None => {
                    Wire::new(*temp_index, DType::Bool);
                    *temp_index += 1;
                    Wire::new(*temp_index - 1, DType::Bool)
                }
            };
            terms.push(Term::function::<Wire<DType>, Wire<DType>, _, _>(
                IType::ConstBool(false),
                [write.clone()],
                vec![],
            ).unwrap());
            enforce(&final_write, terms, write, vec![])
        }

        Rule::expr_primary => {
            if let Some(inner) = expr_pair.into_inner().next() {
                lower_expr_to_terms(inner, wires, var_count, n, final_write, temp_index, terms)
            } else {
                enforce(&final_write, terms, Wire::new(0, DType::Bool), vec![])
            }
        }

        Rule::abs_call => {
            // abs(expr) -> lowered as: let t0 = inner; let neg = 0 - t0; let cond = t0 < 0; result = cond ? neg : t0
            // Find the actual expression child (skip the ABS token and parentheses)
            let mut inner_expr: Option<Pair<Rule>> = None;
            for child in expr_pair.clone().into_inner() {
                match child.as_rule() {
                    Rule::expr
                    | Rule::expr_cond
                    | Rule::expr_or
                    | Rule::expr_and
                    | Rule::expr_cmp
                    | Rule::expr_arith
                    | Rule::expr_term
                    | Rule::expr_factor
                    | Rule::expr_primary
                    | Rule::expr_assign => {
                        inner_expr = Some(child);
                        break;
                    }
                    _ => {}
                }
            }
            let inner = match inner_expr {
                Some(p) => p,
                None => return enforce(&final_write, terms, Wire::new(0, DType::Int), vec![]),
            };
            // lower inner first (produces temps/terms as needed)
            let (inner_w, inner_read) =
                lower_expr_to_terms(inner, wires, var_count, n, None, temp_index, terms);

            // If we're writing into a final (primed) wire (init context), and
            // the inner expression is a direct IVAR reference (unprimed idx
            // in [var_count, n)), then the Abs term should read the primed
            // IVAR (ridx + n). Build a mapped read wire for the Abs term but
            // keep `inner_read` unchanged for classification purposes.
            let mut abs_read_for_term = vec![];
            let i = inner_w.id();
            let dt = inner_w.dtype();
            if i < n && i >= var_count {
                abs_read_for_term.push(Wire::new(i + n, *dt));
            } else {
                abs_read_for_term.push(Wire::new(i, *dt));
            }

            let write = match &final_write {
                Some(w) => w.clone(),
                None => {
                    let w = Wire::new(*temp_index, DType::Int);
                    *temp_index += 1;
                    w
                }
            };

            terms.push(Term::function(
                IType::Abs,
                [write.clone()],
                abs_read_for_term.clone(),
            ).unwrap());
            enforce(&final_write, terms, write, inner_read)
        }

        _ => {
            if let Some(inner) = expr_pair.clone().into_inner().next() {
                lower_expr_to_terms(inner, wires, var_count, n, final_write, temp_index, terms)
            } else {
                enforce(&final_write, terms, Wire::new(0, DType::Bool), vec![])
            }
        }
    }
}

// Classify reads from a read-union Vec<Wire> but only consider original wires
// (indices < n). `var_count` splits vars vs ivars.
fn classify_reads_from_wire(
    read: &[Wire<DType>],
    var_count: usize,
    n: usize,
    read_latched: &mut Vec<Wire<DType>>,
    wait_latched: &mut Vec<Wire<DType>>,
) {
    // Helper: append a single wire into target if not present
    fn add_unique(target: &mut Vec<Wire<DType>>, idx: usize, dtype: DType) {
        if !target.iter().any(|w| w.id() == idx) {
            target.push(Wire::new(idx, dtype));
        }
    }

    for wire in read.iter() {
        let ri = wire.id();
        let rdtype = wire.dtype();
        if ri < var_count {
            add_unique(read_latched, ri, *rdtype);
        } else if ri < n {
            add_unique(wait_latched, ri, *rdtype);
        } else {
            // temps and other generated wires are ignored for classification
        }
    }
}

#[derive(Parser)]
#[grammar = "smv.pest"]
pub struct SMVParser;

pub fn parse_smv(input: &str) -> Result<Module<DType, IType>, &'static str> {
    let parsed = SMVParser::parse(Rule::file, input)
        .map_err(|e| {
            eprintln!("Pest parse error: {}", e);
            "parse failed"
        })?
        .next()
        .ok_or("empty parse tree")?;
    build_module(parsed)
}

fn build_module(file_pair: Pair<Rule>) -> Result<Module<DType, IType>, &'static str> {
    // Collect VAR and IVAR separately so we can enforce a canonical ordering
    // when building the final wire vector. This affects index assignment
    // and therefore the inferred extl/intf/prvt layout in Module.
    let mut var_decls: Vec<(String, DType)> = vec![];
    let mut ivar_decls: Vec<(String, DType)> = vec![];
    let mut wires: Vec<(String, usize, DType)> = vec![];
    let mut init_assigns: Vec<(String, Pair<Rule>)> = vec![];
    let mut next_assigns: Vec<(String, Pair<Rule>)> = vec![];
    // collect INVAR expressions (kept for lowering into Terms later)
    let mut invar_exprs: Vec<Pair<Rule>> = vec![];

    // Step 1: parse variables and assignments
    for section in file_pair.into_inner() {
        if section.as_rule() != Rule::module_decl {
            continue;
        }
        for inner in section.into_inner() {
            if inner.as_rule() != Rule::module_body {
                continue;
            }
            for body_item in inner.into_inner() {
                match body_item.as_rule() {
                    Rule::ivar_section => {
                        for decl in body_item
                            .into_inner()
                            .filter(|p| p.as_rule() == Rule::ivar_decl)
                        {
                            let mut decl_iter = decl.into_inner();
                            let name = decl_iter.next().unwrap().as_str().to_string();
                            let dtype_rule = decl_iter.next().unwrap();
                            let dtype = match dtype_rule.as_str() {
                                "boolean" => DType::Bool,
                                "integer" => DType::Int,
                                _ => DType::Bool,
                            };
                            ivar_decls.push((name, dtype));
                        }
                    }

                    Rule::var_section => {
                        for decl in body_item
                            .into_inner()
                            .filter(|p| p.as_rule() == Rule::var_decl)
                        {
                            let mut decl_iter = decl.into_inner();
                            let name = decl_iter.next().unwrap().as_str().to_string();
                            let dtype_rule = decl_iter.next().unwrap();
                            let dtype = match dtype_rule.as_str() {
                                "boolean" => DType::Bool,
                                "integer" => DType::Int,
                                _ => DType::Bool,
                            };
                            var_decls.push((name, dtype));
                        }
                    }

                    Rule::assign_section => {
                        for assign in body_item
                            .into_inner()
                            .filter(|p| p.as_rule() == Rule::assign_stmt)
                        {
                            let mut parts = assign.into_inner();
                            let target_pair = parts.next().unwrap();
                            let expr_pair = parts.next().unwrap();

                            match target_pair.as_rule() {
                                Rule::init_ref => {
                                    let var_name = target_pair
                                        .into_inner()
                                        .find(|p| p.as_rule() == Rule::ident)
                                        .unwrap()
                                        .as_str()
                                        .to_string();
                                    init_assigns.push((var_name, expr_pair));
                                }
                                Rule::next_ref => {
                                    let var_name = target_pair
                                        .into_inner()
                                        .find(|p| p.as_rule() == Rule::ident)
                                        .unwrap()
                                        .as_str()
                                        .to_string();
                                    next_assigns.push((var_name, expr_pair));
                                }
                                Rule::ident => {
                                    // direct assignment treated as init
                                    let var_name = target_pair.as_str().to_string();
                                    init_assigns.push((var_name, expr_pair));
                                }
                                _ => {}
                            }
                        }
                    }
                    Rule::invar_section => {
                        // invar_section := INVAR expr ;
                        for expr in body_item.into_inner().filter(|p| p.as_rule() == Rule::expr) {
                            invar_exprs.push(expr.clone());
                        }
                    }
                    Rule::init_section => {
                        return Err("INIT sections not supported; use ASSIGN with init(x) := ...");
                    }
                    Rule::trans_section => {
                        return Err("TRANS sections not supported");
                    }
                    _ => {}
                }
            }
        }
    }

    // Assemble wires in a canonical ordering so indices are stable and
    // so Module inference of extl/intf/prvt matches expectations.
    // We place VAR declarations first, then IVAR declarations.
    for (name, dtype) in var_decls.iter() {
        let index = wires.len();
        wires.push((name.clone(), index, *dtype));
    }
    for (name, dtype) in ivar_decls.iter() {
        let index = wires.len();
        wires.push((name.clone(), index, *dtype));
    }

    let n = wires.len();

    // Step 2: latched and next wires
    let mut latched = vec![];
    let mut next_wire = vec![];
    for (_, i, dtype) in &wires {
        latched.push(Wire::new(*i, *dtype));
        next_wire.push(Wire::new(i + n, *dtype));
    }

    // Step 3: (was previously used) IVAR indices detection is implicit below

    // Step 4: build per-variable init and update Terms.
    // We'll write init/update to the 'next' wires (offset by n) and read
    // from latched wires (and IVARs) as reported by build_expr with offset=0.
    let var_count = var_decls.len();

    let mut init_terms: Vec<Term<DType, IType>> = vec![];
    let mut update_terms: Vec<Term<DType, IType>> = vec![];

    let mut ctrl_latched = vec![];
    let mut wait_latched = vec![];
    let mut read_latched = vec![];

    // Pre-populate wait_latched from IVAR declarations (they live after vars)
    for (i, (_name, dtype)) in ivar_decls.iter().enumerate() {
        let index = var_count + i;
        if !wait_latched.iter().any(|w: &Wire<DType>| w.id() == index) {
            wait_latched.push(Wire::new(index, *dtype));
        }
    }

    // temporaries start at 2*n to avoid colliding with latched/next ranges
    let mut temp_index: usize = 2 * n;
    for i in 0..var_count {
        let (name, idx, dtype) = &wires[i];

        // INIT
        if let Some((_, expr_pair)) = init_assigns.iter().find(|(n, _)| n == name) {
            let write = Wire::new(idx + n, *dtype);
            // detect simple single-leaf RHS (drill down single-child chains)
            let mut leaf = expr_pair.clone();
            loop {
                let mut inner_iter = leaf.clone().into_inner();
                match inner_iter.next() {
                    Some(next) if inner_iter.next().is_none() => {
                        leaf = next;
                        continue;
                    }
                    _ => {}
                }
                break;
            }

            match leaf.as_rule() {
                Rule::number => {
                    let sval = leaf.as_str();
                    let parsed = sval.parse::<i64>().expect("invalid numeric literal");
                    init_terms.push(Term::function::<Wire<DType>, Wire<DType>, _, _>(
                        IType::ConstInt(parsed),
                        [write.clone()],
                        vec![],
                    ).unwrap());
                }
                Rule::ident => {
                    let nm = leaf.as_str().to_string();
                    if let Some((_, ridx, rdtype)) = wires.iter().find(|(n, _, _)| n == &nm) {
                        // If the RHS is an IVAR (index >= var_count) then the
                        // semantic init should use the IVAR's primed twin as the
                        // source value (so the init term reads from ridx + n).
                        // However, for classification of which original
                        // variables are read (to compute wait_latched), we must
                        // report the unprimed IVAR index. So emit an Assign
                        // reading from the primed wire but classify using the
                        // original unprimed wire.
                        if *ridx < var_count {
                            let read = Wire::new(*ridx, *rdtype);
                            init_terms.push(Term::function(IType::Assign, [write.clone()], vec![read.clone()]).unwrap());
                            classify_reads_from_wire(
                                &[read],
                                var_count,
                                n,
                                &mut read_latched,
                                &mut wait_latched,
                            );
                        } else {
                            let unprimed = Wire::new(*ridx, *rdtype);
                            let primed = Wire::new(*ridx + n, *rdtype);
                            init_terms.push(Term::function(
                                IType::Assign,
                                [write.clone()],
                                vec![primed.clone()],
                            ).unwrap());
                            // classify based on the unprimed IVAR index so the
                            // module's wait_latched set includes this IVAR.
                            classify_reads_from_wire(
                                &[unprimed],
                                var_count,
                                n,
                                &mut read_latched,
                                &mut wait_latched,
                            );
                        }
                    } else {
                        let (_final_write, read_union) = lower_expr_to_terms(
                            expr_pair.clone(),
                            &wires,
                            var_count,
                            n,
                            Some(write.clone()),
                            &mut temp_index,
                            &mut init_terms,
                        );
                        classify_reads_from_wire(
                            &read_union,
                            var_count,
                            n,
                            &mut read_latched,
                            &mut wait_latched,
                        );
                    }
                }
                Rule::abs_call => {
                    // abs(inner) where inner may be ident/number/etc. If inner is ident, treat similarly
                    if let Some(inner) = leaf.clone().into_inner().next()
                        && inner.as_rule() == Rule::ident
                    {
                        let nm = inner.as_str().to_string();
                        if let Some((_, ridx, rdtype)) = wires.iter().find(|(n, _, _)| n == &nm) {
                            if *ridx < var_count {
                                let read = Wire::new(*ridx, *rdtype);
                                init_terms.push(Term::function(
                                    IType::Assign,
                                    [write.clone()],
                                    vec![read.clone()],
                                ).unwrap());
                                classify_reads_from_wire(
                                    &[read],
                                    var_count,
                                    n,
                                    &mut read_latched,
                                    &mut wait_latched,
                                );
                            } else {
                                let unprimed = Wire::new(*ridx, *rdtype);
                                let primed = Wire::new(*ridx + n, *rdtype);
                                init_terms.push(Term::function(
                                    IType::Assign,
                                    [write.clone()],
                                    vec![primed.clone()],
                                ).unwrap());
                                classify_reads_from_wire(
                                    &[unprimed],
                                    var_count,
                                    n,
                                    &mut read_latched,
                                    &mut wait_latched,
                                );
                            }
                            continue;
                        }
                    }
                    // fallback: explicitly expand abs(inner) here so we can
                    // ensure the final Cond writes directly into the primed
                    // `write` wire and that the term reads use primed IVAR
                    // indices while classification uses the original (unprimed)
                    // inner reads. This keeps the Module representation
                    // deterministic and avoids leaving an Abs term in a temp.
                    // Lower the inner expression first (no final_write so it
                    // may produce temps).
                    // Find the actual inner expression inside the abs(...) pair
                    let mut inner_expr: Option<Pair<Rule>> = None;
                    for child in leaf.clone().into_inner() {
                        match child.as_rule() {
                            Rule::expr
                            | Rule::expr_cond
                            | Rule::expr_or
                            | Rule::expr_and
                            | Rule::expr_cmp
                            | Rule::expr_arith
                            | Rule::expr_term
                            | Rule::expr_factor
                            | Rule::expr_primary
                            | Rule::expr_assign => {
                                inner_expr = Some(child);
                                break;
                            }
                            _ => {}
                        }
                    }

                    let (inner_w, inner_read) = if let Some(p) = inner_expr {
                        lower_expr_to_terms(
                            p,
                            &wires,
                            var_count,
                            n,
                            None,
                            &mut temp_index,
                            &mut init_terms,
                        )
                    } else {
                        // fallback to lowering the full leaf if we couldn't find a child
                        lower_expr_to_terms(
                            leaf.clone(),
                            &wires,
                            var_count,
                            n,
                            None,
                            &mut temp_index,
                            &mut init_terms,
                        )
                    };

                    // build a mapped view of the inner wire for term reads:
                    // if an inner entry is an IVAR (var_count <= i < n) then
                    // for init we must read its primed twin (i + n). Other
                    // indices are used as-is.
                    let mut mapped_inner = vec![];
                    let i = inner_w.id();
                    let dt = inner_w.dtype();
                    if i < n && i >= var_count {
                        mapped_inner.push(Wire::new(i + n, *dt));
                    } else {
                        mapped_inner.push(Wire::new(i, *dt));
                    }

                    // Emit an Abs term that writes into a temporary, then an
                    // Assign from that temporary into the primed `write` wire.
                    // The Abs term's read should reference primed IVAR indices
                    // (mapped_inner) while classification uses the original
                    // inner_read (unprimed).
                    let temp_w = Wire::new(temp_index, DType::Int);
                    temp_index += 1;
                    init_terms.push(Term::function(IType::Abs, [temp_w.clone()], mapped_inner.clone()).unwrap());

                    // Move the Abs result into the variable's primed next wire
                    init_terms.push(Term::function(IType::Assign, [write.clone()], vec![temp_w.clone()]).unwrap());

                    // classification: report original reads from inner_read
                    classify_reads_from_wire(
                        &inner_read,
                        var_count,
                        n,
                        &mut read_latched,
                        &mut wait_latched,
                    );
                }
                _ => {
                    // fallback to full lowering
                    let (_final_write, read_union) = lower_expr_to_terms(
                        expr_pair.clone(),
                        &wires,
                        var_count,
                        n,
                        Some(write.clone()),
                        &mut temp_index,
                        &mut init_terms,
                    );
                    classify_reads_from_wire(
                        &read_union,
                        var_count,
                        n,
                        &mut read_latched,
                        &mut wait_latched,
                    );
                }
            }
        } else {
            // default init: constant zero / false to next wire
            let default_val = match dtype {
                DType::Bool => IType::ConstBool(false),
                DType::Int => IType::ConstInt(0),
            };
            let write = Wire::new(idx + n, *dtype);
            init_terms.push(Term::function::<Wire<DType>, Wire<DType>, _, _>(default_val, [write], vec![]).unwrap());
        }

        // UPDATE
        if let Some((_, expr_pair)) = next_assigns.iter().find(|(n, _)| n == name) {
            // include this var in ctrl (latched) — avoid duplicates
            if !ctrl_latched.iter().any(|w: &Wire<DType>| w.id() == *idx) {
                ctrl_latched.push(Wire::new(*idx, *dtype));
            }

            // detect simple single-leaf RHS (drill down single-child chains)
            let mut leaf = expr_pair.clone();
            loop {
                let mut inner_iter = leaf.clone().into_inner();
                match inner_iter.next() {
                    Some(next) if inner_iter.next().is_none() => {
                        leaf = next;
                        continue;
                    }
                    _ => {}
                }
                break;
            }

            let write = Wire::new(idx + n, *dtype);
            match leaf.as_rule() {
                Rule::number => {
                    let sval = leaf.as_str();
                    let parsed = sval.parse::<i64>().expect("invalid numeric literal");
                    update_terms.push(Term::function::<Wire<DType>, Wire<DType>, _, _>(
                        IType::ConstInt(parsed),
                        [write.clone()],
                        vec![],
                    ).unwrap());
                }
                Rule::ident => {
                    let nm = leaf.as_str().to_string();
                    if let Some((_, ridx, rdtype)) = wires.iter().find(|(n, _, _)| n == &nm) {
                        let read = Wire::new(*ridx, *rdtype);
                        update_terms.push(Term::function(IType::Assign, [write.clone()], vec![read.clone()]).unwrap());
                        classify_reads_from_wire(
                            &[read],
                            var_count,
                            n,
                            &mut read_latched,
                            &mut wait_latched,
                        );
                    } else {
                        let (_final_write, read_union) = lower_expr_to_terms(
                            expr_pair.clone(),
                            &wires,
                            var_count,
                            n,
                            Some(write.clone()),
                            &mut temp_index,
                            &mut update_terms,
                        );
                        classify_reads_from_wire(
                            &read_union,
                            var_count,
                            n,
                            &mut read_latched,
                            &mut wait_latched,
                        );
                    }
                }
                Rule::abs_call => {
                    if let Some(inner) = leaf.clone().into_inner().next()
                        && inner.as_rule() == Rule::ident
                    {
                        let nm = inner.as_str().to_string();
                        if let Some((_, ridx, rdtype)) = wires.iter().find(|(n, _, _)| n == &nm) {
                            let read = Wire::new(*ridx, *rdtype);
                            update_terms.push(Term::function(
                                IType::Assign,
                                [write.clone()],
                                vec![read.clone()],
                            ).unwrap());
                            classify_reads_from_wire(
                                &[read],
                                var_count,
                                n,
                                &mut read_latched,
                                &mut wait_latched,
                            );
                            // done
                            continue;
                        }
                    }
                    let (_final_write, read_union) = lower_expr_to_terms(
                        expr_pair.clone(),
                        &wires,
                        var_count,
                        n,
                        Some(write.clone()),
                        &mut temp_index,
                        &mut update_terms,
                    );
                    classify_reads_from_wire(
                        &read_union,
                        var_count,
                        n,
                        &mut read_latched,
                        &mut wait_latched,
                    );
                }
                _ => {
                    let (_final_write, read_union) = lower_expr_to_terms(
                        expr_pair.clone(),
                        &wires,
                        var_count,
                        n,
                        Some(write.clone()),
                        &mut temp_index,
                        &mut update_terms,
                    );
                    classify_reads_from_wire(
                        &read_union,
                        var_count,
                        n,
                        &mut read_latched,
                        &mut wait_latched,
                    );
                }
            }
        } else {
            // default update: next(var) := var
            if !ctrl_latched.iter().any(|w: &Wire<DType>| w.id() == *idx) {
                ctrl_latched.push(Wire::new(*idx, *dtype));
            }
            let write = Wire::new(idx + n, *dtype);
            let read = Wire::new(*idx, *dtype);
            update_terms.push(Term::function(IType::Assign, [write], vec![read]).unwrap());
        }
    }

    // Step 5: create single Atom for the module using next-twinned ctrl/wait
    let latched_start = latched.first().map(|w| w.id()).unwrap_or(0);
    let next_start = next_wire.first().map(|w| w.id()).unwrap_or(0);
    let offset: isize = (next_start as isize) - (latched_start as isize);
    let ctrl_next = twin(&ctrl_latched, offset).unwrap();
    let wait_next = twin(&wait_latched, offset).unwrap();
    let atom = Atom::sequential(
        ctrl_latched.iter().chain(wait_latched.iter()),
        ctrl_next.iter().chain(wait_next.iter()),
        init_terms,
        update_terms,
    ).unwrap();

    // Use the Module::new constructor which will infer extl/intf/prvt
    // from the provided observable wires and atoms.
    let obs_pairs: Vec<[Wire<DType>; 2]> = latched.iter().zip(next_wire.iter())
        .map(|(l, n)| [l.clone(), n.clone()])
        .collect();
    Module::new(
        obs_pairs,
        std::iter::empty::<[Wire<DType>; 2]>(),
        vec![atom],
    )
}

fn build_expr(
    expr_pair: Pair<Rule>,
    wires: &[(String, usize, DType)],
    offset: usize,
) -> (IType, Wire<DType>, Wire<DType>) {
    match expr_pair.as_rule() {
        // conditional ?: (expr_cond)
        Rule::expr_cond => {
            let mut inner = expr_pair.into_inner();
            let cond_pair = inner.next().unwrap();
            let cond = build_expr(cond_pair, wires, offset);

            if inner.peek().is_some() {
                // consume "?" and parse branches
                inner.next(); // skip '?'
                let then_pair = inner.next().unwrap();
                let _then_b = build_expr(then_pair, wires, offset);
                inner.next(); // skip ':'
                let else_pair = inner.next().unwrap();
                let _else_b = build_expr(else_pair, wires, offset);

                // safely combine reads without borrowing
                let read = cond.2.clone();
                // read tracking disabled
                // read tracking disabled

                (IType::Cond, Wire::new(offset, DType::Bool), read)
            } else {
                cond
            }
        }

        // boolean OR
        Rule::expr_or => {
            let mut inner = expr_pair.into_inner();
            let (left_itype, left_write, left_read) = build_expr(inner.next().unwrap(), wires, offset);
            let mut combined = (left_itype, left_write, left_read);
            while let Some(_next_pair) = inner.next() {
                // next_pair is the OR token, next() is the rhs
                let _rhs = build_expr(inner.next().unwrap(), wires, offset);
                let read = combined.2.clone(); // simplified
                combined = (IType::Or, Wire::new(offset, DType::Bool), read);
            }
            combined
        }

        // boolean AND
        Rule::expr_and => {
            let mut inner = expr_pair.into_inner();
            let (left_itype, left_write, left_read) = build_expr(inner.next().unwrap(), wires, offset);
            let mut combined = (left_itype, left_write, left_read);
            while inner.peek().is_some() {
                let _rhs = build_expr(inner.next().unwrap(), wires, offset);
                let read = combined.2.clone(); // simplified
                combined = (IType::And, Wire::new(offset, DType::Bool), read);
            }
            combined
        }

        // comparison level (expr_cmp): may chain comparisons, but we'll fold left-to-right
        Rule::expr_cmp => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let mut left = build_expr(first, wires, offset);

            while let Some(op) = inner.next() {
                let right_pair = inner.next().unwrap();
                let _right = build_expr(right_pair, wires, offset);
                let read = left.2.clone(); // simplified
                left = match op.as_rule() {
                    Rule::LT => (IType::Lt, Wire::new(offset, DType::Bool), read),
                    Rule::LE => (IType::Le, Wire::new(offset, DType::Bool), read),
                    Rule::GT => (IType::Gt, Wire::new(offset, DType::Bool), read),
                    Rule::GE => (IType::Ge, Wire::new(offset, DType::Bool), read),
                    Rule::EQ => (IType::Eq, Wire::new(offset, DType::Bool), read),
                    _ => left, // shouldn't happen
                };
            }
            left
        }

        // arithmetic (+,-)
        Rule::expr_arith => {
            let mut inner = expr_pair.into_inner();
            let first_term = inner.next().unwrap();
            let mut left = build_expr(first_term, wires, offset);

            while let Some(op) = inner.next() {
                let right_term = inner.next().unwrap();
                let _right = build_expr(right_term, wires, offset);
                let combined_read = left.2.clone(); // simplified

                left = match op.as_rule() {
                    Rule::PLUS => (IType::Add, Wire::new(offset, DType::Int), combined_read),
                    Rule::MINUS => (IType::Sub, Wire::new(offset, DType::Int), combined_read),
                    _ => {
                        left = build_expr(op, wires, offset);
                        continue;
                    }
                };
            }
            left
        }

        // term level (*,/)
        Rule::expr_term => {
            let mut inner = expr_pair.into_inner();
            let first_primary = inner.next().unwrap();
            let mut left = build_expr(first_primary, wires, offset);

            while let Some(op) = inner.next() {
                let right_primary = inner.next().unwrap();
                let _right = build_expr(right_primary, wires, offset);
                let combined_read = left.2.clone(); // simplified

                left = match op.as_rule() {
                    Rule::TIMES => (IType::Mul, Wire::new(offset, DType::Int), combined_read),
                    Rule::DIVIDE => (IType::Div, Wire::new(offset, DType::Int), combined_read),
                    _ => {
                        left = build_expr(op, wires, offset);
                        continue;
                    }
                };
            }
            left
        }

        // factor level: NOT or primary
        Rule::expr_factor => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            if first.as_rule() == Rule::NOT {
                // next item is the factor being negated
                let inner_factor = inner.next().unwrap();
                let (_sub_itype, _, sub_read) = build_expr(inner_factor, wires, offset);
                (IType::Not, Wire::new(offset, DType::Bool), sub_read)
            } else {
                // first is a primary expression
                build_expr(first, wires, offset)
            }
        }

        Rule::expr_assign => {
            let mut inner = expr_pair.into_inner();
            let _target = inner.next().unwrap();
            let value_expr = inner.next().unwrap();
            let rhs = build_expr(value_expr, wires, offset);
            (IType::Assign, Wire::new(0, DType::Bool), rhs.2.clone())
        }

        // primary tokens
        Rule::number => {
            let sval = expr_pair.as_str();
            let parsed = sval.parse::<i64>().expect("invalid numeric literal");
            (IType::ConstInt(parsed), Wire::new(0, DType::Int), Wire::new(0, DType::Int))
        }

        Rule::ident => {
            let name = expr_pair.as_str().to_string();
            if let Some((_, idx, dtype)) = wires.iter().find(|(n, _, _)| n == &name) {
                // Variable references are represented by Assign with the read wire set to the variable
                (IType::Assign, Wire::new(0, DType::Bool), Wire::new(idx + offset, *dtype))
            } else {
                (IType::Assign, Wire::new(0, DType::Bool), Wire::new(0, DType::Bool))
            }
        }

        Rule::TRUE => (IType::ConstBool(false), Wire::new(0, DType::Bool), Wire::new(0, DType::Bool)),
        Rule::FALSE => (IType::ConstBool(false), Wire::new(0, DType::Bool), Wire::new(0, DType::Bool)),

        Rule::expr_primary => {
            // unwrap parenthesized expression or forward to inner
            if let Some(inner) = expr_pair.into_inner().next() {
                build_expr(inner, wires, offset)
            } else {
                (IType::ConstBool(false), Wire::new(0, DType::Bool), Wire::new(0, DType::Bool)) // unreachable fallback
            }
        }

        Rule::abs_call => {
            // abs(expr) — hacky: treat as identity of inner expression
            if let Some(inner) = expr_pair.into_inner().next() {
                build_expr(inner, wires, offset)
            } else {
                (IType::ConstBool(false), Wire::new(0, DType::Bool), Wire::new(0, DType::Bool))
            }
        }

        _ => {
            if let Some(inner) = expr_pair.clone().into_inner().next() {
                build_expr(inner, wires, offset)
            } else {
                panic!("unhandled expr in helper: {:?}", expr_pair.as_rule())
            }
        }
    }
}
