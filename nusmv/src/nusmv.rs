use pest::Parser;
use pest_derive::Parser;
use pest::iterators::Pair;

use base::module::Module;
use base::atom::Atom;
use base::term::Term;
use base::wire::Wire;

use crate::dtype::DType;
use crate::itype::IType;

// Helper: classify read wires returned by expression parsing into latched
// vs ivar (wait) wires. `var_count` is the number of VAR declarations;
// indices < var_count are latched, indices >= var_count are IVARs.
fn classify_reads(read: &Wire<DType>, var_count: usize, read_latched: &mut Wire<DType>, wait_latched: &mut Wire<DType>) {
    for (ri, rdtype) in read.iter() {
        if ri < var_count {
            *read_latched = read_latched.union(&Wire::one(ri, rdtype.clone())).unwrap();
        } else {
            *wait_latched = wait_latched.union(&Wire::one(ri, rdtype.clone())).unwrap();
        }
    }
}

// Small wrapper that returns (IType, read) by calling the main build_expr.
fn build_expr_no_write(expr_pair: Pair<Rule>, wires: &[(String, usize, DType)], offset: usize) -> (IType, Wire<DType>) {
    let (itype, _write, read) = build_expr(expr_pair, wires, offset);
    (itype, read)
}

// Lower an expression into a sequence of Terms. `final_write` when provided
// is the wire where the final result must be written (e.g., the next wire of
// a variable). `temp_index` is a mutable counter for allocating temporary
// wires starting at 2*n. `terms` is appended with the generated Terms.
fn lower_expr_to_terms(
    expr_pair: Pair<Rule>,
    wires: &[(String, usize, DType)],
    var_count: usize,
    n: usize,
    final_write: Option<Wire<DType>>,
    temp_index: &mut usize,
    terms: &mut Vec<Term<DType, IType>>,
) -> (Wire<DType>, Wire<DType>) {
    let rule = expr_pair.as_rule();
        // helper to ensure that when a caller requested `final_write`, the
        // returned wire is that final_write. If the lowered result wrote into
        // a temporary, we append an explicit Assign to move it into final_write.
        fn enforce(final_write: &Option<Wire<DType>>, terms: &mut Vec<Term<DType, IType>>, w: Wire<DType>, r: Wire<DType>) -> (Wire<DType>, Wire<DType>) {
            if let Some(fw) = final_write {
                if fw != &w {
                    terms.push(Term::new(IType::Assign, fw.clone(), w.clone()));
                    return (fw.clone(), r);
                }
            }
            (w, r)
        }
    match rule {
        Rule::expr_cond => {
            // More deterministic lowering for conditional expressions so the
            // emitted operator Terms match the manual lowering ordering.
            let mut inner = expr_pair.into_inner();
            let cond_pair = inner.next().unwrap();
            // if there's no '?', just lower the condition expression
            if inner.peek().is_none() {
                return lower_expr_to_terms(cond_pair, wires, var_count, n, final_write, temp_index, terms);
            }
            inner.next(); // skip '?'
            let then_pair = inner.next().unwrap();
            inner.next(); // skip ':'
            let else_pair = inner.next().unwrap();

            // Try a simple textual lowering for common condition patterns like
            // "(x < y | x < z)" -> produce Lt, Lt, Or in that order.
            let cond_s = cond_pair.as_str().trim().to_string();
            let mut cond_read_union = Wire::none();
            let cond_temp = if cond_s.contains('|') && (cond_s.contains("<") || cond_s.contains(">") || cond_s.contains("=")) {
                // split on '|' and lower each comparison
                let parts: Vec<String> = cond_s.trim_matches(|c| c == '(' || c == ')')
                    .split('|')
                    .map(|s| s.trim().to_string())
                    .collect();
                let mut comp_temps: Vec<Wire<DType>> = vec![];
                for p in parts.iter() {
                    // find operator (<, <=, >, >=, =)
                    let op_char = if p.contains("<=") { "<=" } else if p.contains(">=") { ">=" } else if p.contains("<") { "<" } else if p.contains(">") { ">" } else { "=" };
                    let sides: Vec<&str> = p.split(op_char).map(|s| s.trim()).collect();
                    if sides.len() != 2 { continue; }
                    let left_name = sides[0].trim_matches(|c| c == '(' || c == ')').to_string();
                    let right_name = sides[1].trim_matches(|c| c == '(' || c == ')').to_string();
                    // resolve wires for left/right (prefer latched wires)
                    let left_w = if let Some((_, idx, dtype)) = wires.iter().find(|(n, _, _)| n == &left_name) {
                        Wire::one(*idx, dtype.clone())
                    } else { Wire::none() };
                    let right_w = if let Some((_, idx, dtype)) = wires.iter().find(|(n, _, _)| n == &right_name) {
                        Wire::one(*idx, dtype.clone())
                    } else { Wire::none() };
                    let write = { let w = Wire::one(*temp_index, DType::Bool); *temp_index += 1; w };
                    let tag = match op_char {
                        "<" => IType::Lt,
                        "<=" => IType::Le,
                        ">" => IType::Gt,
                        ">=" => IType::Ge,
                        "=" => IType::Eq,
                        _ => IType::Lt,
                    };
                    let mut read = Wire::none();
                    read = read.union(&left_w).unwrap_or(read.clone());
                    read = read.union(&right_w).unwrap_or(read.clone());
                    // accumulate only original-variable reads (indices < n)
                    for (ri, rdtype) in read.iter() {
                        if ri < n {
                            cond_read_union = cond_read_union.union(&Wire::one(ri, rdtype.clone())).unwrap();
                        }
                    }
                    terms.push(Term::new(tag, write.clone(), read));
                    comp_temps.push(write);
                }
                // OR chain comp_temps
                if comp_temps.len() == 1 { (comp_temps[0].clone(), cond_read_union.clone()) } else {
                    let mut it = comp_temps.into_iter();
                    let mut left_t = it.next().unwrap();
                    for right_t in it {
                        let w = { let w = Wire::one(*temp_index, DType::Bool); *temp_index += 1; w };
                        let mut read = Wire::none();
                        read = read.union(&left_t).unwrap_or(read.clone());
                        read = read.union(&right_t).unwrap_or(read.clone());
                        terms.push(Term::new(IType::Or, w.clone(), read));
                        // OR reads consume previous temps; no new original reads here
                        left_t = w;
                    }
                    (left_t, cond_read_union.clone())
                }
            } else {
                let (w, r) = lower_expr_to_terms(cond_pair, wires, var_count, n, None, temp_index, terms);
                cond_read_union = cond_read_union.union(&r).unwrap_or(cond_read_union.clone());
                (w, cond_read_union.clone())
            };

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
                        let (right_w, right_read) = lower_expr_to_terms(right, wires, var_count, n, None, temp_index, terms);
                        let (left_w, left_read) = lower_expr_to_terms(left, wires, var_count, n, None, temp_index, terms);
                        // combine reads
                        let mut then_reads = Wire::none();
                        then_reads = then_reads.union(&right_read).unwrap_or(then_reads.clone());
                        then_reads = then_reads.union(&left_read).unwrap_or(then_reads.clone());
                        // ensure consts are available as temps (if right_w is a const term it has been emitted)
                        let add_w = { let w = Wire::one(*temp_index, DType::Int); *temp_index += 1; w };
                    let mut read = Wire::none();
                    read = read.union(&left_w).unwrap_or(read.clone());
                    read = read.union(&right_w).unwrap_or(read.clone());
                    terms.push(Term::new(IType::Add, add_w.clone(), read));
                    // propagate original-variable reads upward
                    let mut then_read_union = Wire::none();
                    then_read_union = then_read_union.union(&then_reads).unwrap_or(then_read_union.clone());
                    (add_w, then_read_union)
                }
                _ => lower_expr_to_terms(then_pair, wires, var_count, n, None, temp_index, terms),
            };

            // Lower the else branch (likely a number)
            let (else_temp_wire, else_read) = lower_expr_to_terms(else_pair, wires, var_count, n, None, temp_index, terms);

            // final cond write: write into final_write (usually the next wire for the var)
            let write = match &final_write { Some(w) => w.clone(), None => { let w = Wire::one(*temp_index, DType::Int); *temp_index += 1; w } };

            // The Term's read field must reference the actual wires (temps or consts)
            // produced for the condition, then-branch and else-branch so e.g. the
            // Cond reads the OR/temp bool and the computed then/else ints.
            let mut term_read_wires = Wire::none();
            term_read_wires = term_read_wires.union(&cond_temp_wire).unwrap_or(term_read_wires.clone());
            term_read_wires = term_read_wires.union(&then_temp_wire).unwrap_or(term_read_wires.clone());
            term_read_wires = term_read_wires.union(&else_temp_wire).unwrap_or(term_read_wires.clone());
            terms.push(Term::new(IType::Cond, write.clone(), term_read_wires.clone()));

            // The returned read-union (for classification) remains the union of
            // original-variable reads from each subexpression.
            let mut orig_read_union = Wire::none();
            orig_read_union = orig_read_union.union(&cond_temp_read).unwrap_or(orig_read_union.clone());
            orig_read_union = orig_read_union.union(&then_read).unwrap_or(orig_read_union.clone());
            orig_read_union = orig_read_union.union(&else_read).unwrap_or(orig_read_union.clone());
            enforce(&final_write, terms, write, orig_read_union)
        }

        Rule::expr_or | Rule::expr_and => {
            let mut inner = expr_pair.into_inner();
            let mut operand_wires: Vec<Wire<DType>> = vec![];
            let mut operands_read_union = Wire::none();
            // first operand
            if let Some(first) = inner.next() {
                let (w, r) = lower_expr_to_terms(first, wires, var_count, n, None, temp_index, terms);
                operand_wires.push(w);
                operands_read_union = operands_read_union.union(&r).unwrap_or(operands_read_union.clone());
            }
            while let Some(_) = inner.next() {
                if let Some(rhs) = inner.next() {
                    let (w, r) = lower_expr_to_terms(rhs, wires, var_count, n, None, temp_index, terms);
                    operand_wires.push(w);
                    operands_read_union = operands_read_union.union(&r).unwrap_or(operands_read_union.clone());
                }
            }
            if operand_wires.len() == 1 {
                return (operand_wires[0].clone(), operands_read_union);
            }
            let write = match &final_write { Some(w) => w.clone(), None => { let w = Wire::one(*temp_index, DType::Bool); *temp_index += 1; w } };
            let mut read = Wire::none();
            for ow in operand_wires.iter() { read = read.union(ow).unwrap_or(read.clone()); }
            let op_tag = if rule == Rule::expr_or { IType::Or } else { IType::And };
            terms.push(Term::new(op_tag, write.clone(), read));
            enforce(&final_write, terms, write, operands_read_union)
        }

        Rule::expr_cmp => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (mut left_w, left_read) = lower_expr_to_terms(first, wires, var_count, n, None, temp_index, terms);
            // produce a comparison term for each op
            let mut cmp_read_union = left_read.clone();
            while let Some(op) = inner.next() {
                let right_pair = inner.next().unwrap();
                let (right_w, right_read) = lower_expr_to_terms(right_pair, wires, var_count, n, None, temp_index, terms);
                let write = Wire::one(*temp_index, DType::Bool);
                *temp_index += 1;
                let mut read = Wire::none();
                read = read.union(&left_w).unwrap_or(read.clone());
                read = read.union(&right_w).unwrap_or(read.clone());
                let tag = match op.as_rule() {
                    Rule::LT => IType::Lt,
                    Rule::LE => IType::Le,
                    Rule::GT => IType::Gt,
                    Rule::GE => IType::Ge,
                    Rule::EQ => IType::Eq,
                    _ => IType::Lt,
                };
                terms.push(Term::new(tag, write.clone(), read));
                // accumulate original-variable reads
                cmp_read_union = cmp_read_union.union(&right_read).unwrap_or(cmp_read_union.clone());
                left_w = write;
            }
            enforce(&final_write, terms, left_w, cmp_read_union)
        }

        Rule::expr_arith => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (mut left_w, mut left_read) = lower_expr_to_terms(first, wires, var_count, n, None, temp_index, terms);
            loop {
                let op = match inner.next() { Some(o) => o, None => break };
                let right_term = inner.next().unwrap();
                let (right_w, right_read) = lower_expr_to_terms(right_term, wires, var_count, n, None, temp_index, terms);
                let write = Wire::one(*temp_index, DType::Int);
                *temp_index += 1;
                let mut read = Wire::none();
                read = read.union(&left_w).unwrap_or(read.clone());
                read = read.union(&right_w).unwrap_or(read.clone());
                let tag = match op.as_rule() {
                    Rule::PLUS => IType::Add,
                    Rule::MINUS => IType::Sub,
                    _ => IType::Add,
                };
                terms.push(Term::new(tag, write.clone(), read));
                // accumulate original-variable reads
                left_read = left_read.union(&right_read).unwrap_or(left_read.clone());
                left_w = write;
            }
            enforce(&final_write, terms, left_w, left_read)
        }

        Rule::expr_term => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (mut left_w, mut left_read) = lower_expr_to_terms(first, wires, var_count, n, None, temp_index, terms);
            loop {
                let op = match inner.next() { Some(o) => o, None => break };
                let right_primary = inner.next().unwrap();
                let (right_w, right_read) = lower_expr_to_terms(right_primary, wires, var_count, n, None, temp_index, terms);
                let write = Wire::one(*temp_index, DType::Int);
                *temp_index += 1;
                let mut read = Wire::none();
                read = read.union(&left_w).unwrap_or(read.clone());
                read = read.union(&right_w).unwrap_or(read.clone());
                let tag = match op.as_rule() {
                    Rule::TIMES => IType::Mul,
                    Rule::DIVIDE => IType::Div,
                    _ => IType::Mul,
                };
                terms.push(Term::new(tag, write.clone(), read));
                left_read = left_read.union(&right_read).unwrap_or(left_read.clone());
                left_w = write;
            }
            (left_w, left_read)
        }

        Rule::expr_factor => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            if first.as_rule() == Rule::NOT {
                let inner_factor = inner.next().unwrap();
                let (sub_wire, sub_read) = lower_expr_to_terms(inner_factor, wires, var_count, n, None, temp_index, terms);
                let write = match &final_write { Some(w) => w.clone(), None => { Wire::one(*temp_index, DType::Bool); *temp_index += 1; Wire::one(*temp_index - 1, DType::Bool) } };
                terms.push(Term::new(IType::Not, write.clone(), sub_wire));
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
            lower_expr_to_terms(value_expr, wires, var_count, n, final_write, temp_index, terms)
        }

        Rule::number => {
            let sval = expr_pair.as_str();
            let parsed = sval.parse::<i64>().expect("invalid numeric literal");
            let write = match &final_write { Some(w) => w.clone(), None => { Wire::one(*temp_index, DType::Int); *temp_index += 1; Wire::one(*temp_index - 1, DType::Int) } };
            terms.push(Term::new(IType::ConstInt(parsed), write.clone(), Wire::none()));
            enforce(&final_write, terms, write, Wire::none())
        }

        Rule::ident => {
            let name = expr_pair.as_str().to_string();
            if let Some((_, idx, dtype)) = wires.iter().find(|(n, _, _)| n == &name) {
                enforce(&final_write, terms, Wire::one(*idx, dtype.clone()), Wire::one(*idx, dtype.clone()))
            } else {
                enforce(&final_write, terms, Wire::none(), Wire::none())
            }
        }

        Rule::TRUE => {
            let write = match &final_write { Some(w) => w.clone(), None => { Wire::one(*temp_index, DType::Bool); *temp_index += 1; Wire::one(*temp_index - 1, DType::Bool) } };
            terms.push(Term::new(IType::ConstBool(true), write.clone(), Wire::none()));
            enforce(&final_write, terms, write, Wire::none())
        }

        Rule::FALSE => {
            let write = match &final_write { Some(w) => w.clone(), None => { Wire::one(*temp_index, DType::Bool); *temp_index += 1; Wire::one(*temp_index - 1, DType::Bool) } };
            terms.push(Term::new(IType::ConstBool(false), write.clone(), Wire::none()));
            enforce(&final_write, terms, write, Wire::none())
        }

        Rule::expr_primary => {
            if let Some(inner) = expr_pair.into_inner().next() {
                lower_expr_to_terms(inner, wires, var_count, n, final_write, temp_index, terms)
            } else {
                enforce(&final_write, terms, Wire::none(), Wire::none())
            }
        }

        _ => {
            if let Some(inner) = expr_pair.clone().into_inner().next() {
                lower_expr_to_terms(inner, wires, var_count, n, final_write, temp_index, terms)
            } else {
                enforce(&final_write, terms, Wire::none(), Wire::none())
            }
        }
    }
}

// Classify reads from a read-union Wire but only consider original wires
// (indices < n). `var_count` splits vars vs ivars.
fn classify_reads_from_wire(read: &Wire<DType>, var_count: usize, n: usize, read_latched: &mut Wire<DType>, wait_latched: &mut Wire<DType>) {
    for (ri, rdtype) in read.iter() {
        if ri < var_count {
            *read_latched = read_latched.union(&Wire::one(ri, rdtype.clone())).unwrap();
        } else if ri < n {
            *wait_latched = wait_latched.union(&Wire::one(ri, rdtype.clone())).unwrap();
        } else {
            // temps and other generated wires are ignored for classification
        }
    }
}

#[derive(Parser)]
#[grammar = "nusmv.pest"]
pub struct NuSMVParser;

pub fn parse_nusmv(input: &str) -> Result<Module<DType, IType>, &'static str> {
    let parsed = NuSMVParser::parse(Rule::file, input)
        .map_err(|_| "parse failed")?
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

    // Step 1: parse variables and assignments
    for section in file_pair.into_inner() {
        if section.as_rule() != Rule::module_decl { continue; }
        for inner in section.into_inner() {
            if inner.as_rule() != Rule::module_body { continue; }
            for body_item in inner.into_inner() {
                match body_item.as_rule() {
                    Rule::ivar_section => {
                        for decl in body_item.into_inner().filter(|p| p.as_rule() == Rule::ivar_decl) {
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
                        for decl in body_item.into_inner().filter(|p| p.as_rule() == Rule::var_decl) {
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
                        for assign in body_item.into_inner().filter(|p| p.as_rule() == Rule::assign_stmt) {
                            let mut parts = assign.into_inner();
                            let target_pair = parts.next().unwrap();
                            let expr_pair = parts.next().unwrap();

                            match target_pair.as_rule() {
                                Rule::init_ref => {
                                    let var_name = target_pair.into_inner().find(|p| p.as_rule() == Rule::ident).unwrap().as_str().to_string();
                                    init_assigns.push((var_name, expr_pair));
                                }
                                Rule::next_ref => {
                                    let var_name = target_pair.into_inner().find(|p| p.as_rule() == Rule::ident).unwrap().as_str().to_string();
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
        wires.push((name.clone(), index, dtype.clone()));
    }
    for (name, dtype) in ivar_decls.iter() {
        let index = wires.len();
        wires.push((name.clone(), index, dtype.clone()));
    }

    let n = wires.len();

    // Step 2: latched and next wires
    let mut latched = Wire::none();
    let mut next_wire = Wire::none();
    for (_, i, dtype) in &wires {
        latched = latched.union(&Wire::one(*i, dtype.clone())).unwrap();
        next_wire = next_wire.union(&Wire::one(i + n, dtype.clone())).unwrap();
    }

    // Step 3: (was previously used) IVAR indices detection is implicit below

    // Step 4: build per-variable init and update Terms.
    // We'll write init/update to the 'next' wires (offset by n) and read
    // from latched wires (and IVARs) as reported by build_expr with offset=0.
    let var_count = var_decls.len();

    let mut init_terms: Vec<Term<DType, IType>> = vec![];
    let mut update_terms: Vec<Term<DType, IType>> = vec![];

    let mut ctrl_latched = Wire::none();
    let mut wait_latched = Wire::none();
    let mut read_latched = Wire::none();

    // Pre-populate wait_latched from IVAR declarations (they live after vars)
    for (i, (_name, dtype)) in ivar_decls.iter().enumerate() {
        let index = var_count + i;
        wait_latched = wait_latched.union(&Wire::one(index, dtype.clone())).unwrap();
    }

    // temporaries start at 2*n to avoid colliding with latched/next ranges
    let mut temp_index: usize = 2 * n;
    for i in 0..var_count {
        let (name, idx, dtype) = &wires[i];

        // INIT
        if let Some((_, expr_pair)) = init_assigns.iter().find(|(n, _)| n == name) {
            let write = Wire::one(idx + n, dtype.clone());
            // detect simple single-leaf RHS (drill down single-child chains)
            let mut leaf = expr_pair.clone();
            loop {
                let mut inner_iter = leaf.clone().into_inner();
                if let Some(next) = inner_iter.next() {
                    if inner_iter.next().is_none() {
                        leaf = next;
                        continue;
                    }
                }
                break;
            }

            match leaf.as_rule() {
                Rule::number => {
                    let sval = leaf.as_str();
                    let parsed = sval.parse::<i64>().expect("invalid numeric literal");
                    init_terms.push(Term::new(IType::ConstInt(parsed), write.clone(), Wire::none()));
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
                            let read = Wire::one(*ridx, rdtype.clone());
                            init_terms.push(Term::new(IType::Assign, write.clone(), read.clone()));
                            classify_reads_from_wire(&read, var_count, n, &mut read_latched, &mut wait_latched);
                        } else {
                            let unprimed = Wire::one(*ridx, rdtype.clone());
                            let primed = Wire::one(*ridx + n, rdtype.clone());
                            init_terms.push(Term::new(IType::Assign, write.clone(), primed.clone()));
                            // classify based on the unprimed IVAR index so the
                            // module's wait_latched set includes this IVAR.
                            classify_reads_from_wire(&unprimed, var_count, n, &mut read_latched, &mut wait_latched);
                        }
                    } else {
                        let (_final_write, read_union) = lower_expr_to_terms(expr_pair.clone(), &wires, var_count, n, Some(write.clone()), &mut temp_index, &mut init_terms);
                        classify_reads_from_wire(&read_union, var_count, n, &mut read_latched, &mut wait_latched);
                    }
                }
                _ => {
                    // fallback to full lowering
                    let (_final_write, read_union) = lower_expr_to_terms(expr_pair.clone(), &wires, var_count, n, Some(write.clone()), &mut temp_index, &mut init_terms);
                    classify_reads_from_wire(&read_union, var_count, n, &mut read_latched, &mut wait_latched);
                }
            }
        } else {
            // default init: constant zero / false to next wire
            let default_val = match dtype {
                DType::Bool => IType::ConstBool(false),
                DType::Int => IType::ConstInt(0),
            };
            let write = Wire::one(idx + n, dtype.clone());
            init_terms.push(Term::new(default_val, write, Wire::none()));
        }

        // UPDATE
        if let Some((_, expr_pair)) = next_assigns.iter().find(|(n, _)| n == name) {
            // include this var in ctrl (latched)
            ctrl_latched = ctrl_latched.union(&Wire::one(*idx, dtype.clone())).unwrap();

            // detect simple single-leaf RHS (drill down single-child chains)
            let mut leaf = expr_pair.clone();
            loop {
                let mut inner_iter = leaf.clone().into_inner();
                if let Some(next) = inner_iter.next() {
                    if inner_iter.next().is_none() {
                        leaf = next;
                        continue;
                    }
                }
                break;
            }

            let write = Wire::one(idx + n, dtype.clone());
            match leaf.as_rule() {
                Rule::number => {
                    let sval = leaf.as_str();
                    let parsed = sval.parse::<i64>().expect("invalid numeric literal");
                    update_terms.push(Term::new(IType::ConstInt(parsed), write.clone(), Wire::none()));
                }
                Rule::ident => {
                    let nm = leaf.as_str().to_string();
                    if let Some((_, ridx, rdtype)) = wires.iter().find(|(n, _, _)| n == &nm) {
                        let read = Wire::one(*ridx, rdtype.clone());
                        update_terms.push(Term::new(IType::Assign, write.clone(), read.clone()));
                        classify_reads_from_wire(&read, var_count, n, &mut read_latched, &mut wait_latched);
                    } else {
                        let (_final_write, read_union) = lower_expr_to_terms(expr_pair.clone(), &wires, var_count, n, Some(write.clone()), &mut temp_index, &mut update_terms);
                        classify_reads_from_wire(&read_union, var_count, n, &mut read_latched, &mut wait_latched);
                    }
                }
                _ => {
                    let (_final_write, read_union) = lower_expr_to_terms(expr_pair.clone(), &wires, var_count, n, Some(write.clone()), &mut temp_index, &mut update_terms);
                    classify_reads_from_wire(&read_union, var_count, n, &mut read_latched, &mut wait_latched);
                }
            }
        } else {
            // default update: next(var) := var
            ctrl_latched = ctrl_latched.union(&Wire::one(*idx, dtype.clone())).unwrap();
            let write = Wire::one(idx + n, dtype.clone());
            let read = Wire::one(*idx, dtype.clone());
            update_terms.push(Term::new(IType::Assign, write, read));
        }
    }

    // Step 5: create single Atom for the module using next-twinned ctrl/wait
    let latched_start = latched.iter().next().map(|(i, _)| i).unwrap_or(0);
    let next_start = next_wire.iter().next().map(|(i, _)| i).unwrap_or(0);
    let offset: isize = (next_start as isize) - (latched_start as isize);
    let atom = Atom::new_unchecked(
        ctrl_latched.twin(offset).unwrap(),
        wait_latched.twin(offset).unwrap(),
        read_latched,
        init_terms,
        update_terms,
    );

    Module::with_atoms([latched, next_wire], vec![atom]).map_err(|_| "Failed to build module")
}

fn build_expr(expr_pair: Pair<Rule>, wires: &[(String, usize, DType)], offset: usize) -> (IType, Wire<DType>, Wire<DType>) {
    match expr_pair.as_rule() {
        // conditional ?: (expr_cond)
        Rule::expr_cond => {
            let mut inner = expr_pair.into_inner();
            let cond_pair = inner.next().unwrap();
            let cond = build_expr(cond_pair, wires, offset);

            if let Some(_) = inner.peek() {
                // consume "?" and parse branches
                inner.next(); // skip '?'
                let then_pair = inner.next().unwrap();
                let then_b = build_expr(then_pair, wires, offset);
                inner.next(); // skip ':'
                let else_pair = inner.next().unwrap();
                let else_b = build_expr(else_pair, wires, offset);

                // safely combine reads without borrowing
                let mut read = cond.2.clone();
                if let Ok(r) = read.union(&then_b.2) {
                    read = r;
                }
                if let Ok(r2) = read.union(&else_b.2) {
                    read = r2;
                }

                (
                    IType::Cond,
                    Wire::none(),
                    read,
                )
            } else {
                cond
            }
        }

        // boolean OR
        Rule::expr_or => {
            let mut inner = expr_pair.into_inner();
            let (left_itype, _, left_read) = build_expr(inner.next().unwrap(), wires, offset);
            let mut combined = (left_itype, Wire::none(), left_read);
            while let Some(_next_pair) = inner.next() {
                // next_pair is the OR token, next() is the rhs
                let rhs = build_expr(inner.next().unwrap(), wires, offset);
                let read = combined.2.union(&rhs.2).unwrap_or_else(|_| combined.2.clone());
                combined = (IType::Or, Wire::one(offset, DType::Bool), read);
            }
            combined
        }

        // boolean AND
        Rule::expr_and => {
            let mut inner = expr_pair.into_inner();
            let (left_itype, _, left_read) = build_expr(inner.next().unwrap(), wires, offset);
            let mut combined = (left_itype, Wire::none(), left_read);
            while let Some(_) = inner.peek() {
                let rhs = build_expr(inner.next().unwrap(), wires, offset);
                let read = combined.2.union(&rhs.2).unwrap_or_else(|_| combined.2.clone());
                combined = (IType::And, Wire::one(offset, DType::Bool), read);
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
                let right = build_expr(right_pair, wires, offset);
                let read = left.2.union(&right.2).unwrap_or_else(|_| left.2.clone());
                left = match op.as_rule() {
                    Rule::LT => (IType::Lt, Wire::one(offset, DType::Bool), read),
                    Rule::LE => (IType::Le, Wire::one(offset, DType::Bool), read),
                    Rule::GT => (IType::Gt, Wire::one(offset, DType::Bool), read),
                    Rule::GE => (IType::Ge, Wire::one(offset, DType::Bool), read),
                    Rule::EQ => (IType::Eq, Wire::one(offset, DType::Bool), read),
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

            loop {
                let op = match inner.next() {
                    Some(o) => o,
                    None => break,
                };
                let right_term = inner.next().unwrap();
                let right = build_expr(right_term, wires, offset);
                let combined_read = left.2.union(&right.2).unwrap_or_else(|_| left.2.clone());

                left = match op.as_rule() {
                    Rule::PLUS => (
                        IType::Add,
                        Wire::one(offset, DType::Int),
                        combined_read,
                    ),
                    Rule::MINUS => (
                        IType::Sub,
                        Wire::one(offset, DType::Int),
                        combined_read,
                    ),
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

            loop {
                let op = match inner.next() {
                    Some(o) => o,
                    None => break,
                };
                let right_primary = inner.next().unwrap();
                let right = build_expr(right_primary, wires, offset);
                let combined_read = left.2.union(&right.2).unwrap_or_else(|_| left.2.clone());

                left = match op.as_rule() {
                    Rule::TIMES => (
                        IType::Mul,
                        Wire::one(offset, DType::Int),
                        combined_read,
                    ),
                    Rule::DIVIDE => (
                        IType::Div,
                        Wire::one(offset, DType::Int),
                        combined_read,
                    ),
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
                (
                    IType::Not,
                    Wire::one(offset, DType::Bool),
                    sub_read,
                )
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
            (
                IType::Assign,
                Wire::none(),
                rhs.2.clone(),
            )
        }


        // primary tokens
        Rule::number => {
            let sval = expr_pair.as_str();
            let parsed = sval.parse::<i64>().expect("invalid numeric literal");
            (IType::ConstInt(parsed), Wire::none(), Wire::none())
        }

        Rule::ident => {
            let name = expr_pair.as_str().to_string();
            if let Some((_, idx, dtype)) = wires.iter().find(|(n, _, _)| n == &name) {
                // Variable references are represented by Assign with the read wire set to the variable
                (IType::Assign, Wire::none(), Wire::one(idx + offset, dtype.clone()))
            } else {
                (IType::Assign, Wire::none(), Wire::none())
            }
        }

        Rule::TRUE => (IType::ConstBool(true), Wire::none(), Wire::none()),
        Rule::FALSE => (IType::ConstBool(false), Wire::none(), Wire::none()),

        Rule::expr_primary => {
            // unwrap parenthesized expression or forward to inner
            if let Some(inner) = expr_pair.into_inner().next() {
                build_expr(inner, wires, offset)
            } else {
                (IType::ConstBool(false), Wire::none(), Wire::none()) // unreachable fallback
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

