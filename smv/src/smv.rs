use std::collections::{HashMap, HashSet};

use base::module::Module;
use base::term::Term;
use base::wire::Wire;
use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;

use crate::dtype::DType;
use crate::itype::IType;

/// Result of parsing an SMV file. Contains the reactive module plus any
/// constraint expressions from standalone INIT/TRANS/INVAR sections.
pub struct ParseResult {
    pub module: Module<DType, IType>,
    pub init_constraints: Vec<Term<DType, IType>>,
    pub trans_constraints: Vec<Term<DType, IType>>,
    pub invar_constraints: Vec<Term<DType, IType>>,
}

/// Context threaded through expression lowering to support DEFINE expansion
/// and enum value resolution.
struct LowerCtx<'a> {
    wires: &'a [(String, usize, DType)],
    var_count: usize,
    n: usize,
    defines: &'a HashMap<String, Pair<'a, Rule>>,
    enum_values: &'a HashMap<String, i64>,
    expanding: HashSet<String>,
}

// Lower an expression into a sequence of Terms.
fn lower_expr_to_terms(
    expr_pair: Pair<Rule>,
    ctx: &mut LowerCtx,
    final_write: Option<Wire<DType>>,
    temp_index: &mut usize,
    terms: &mut Vec<Term<DType, IType>>,
) -> (Wire<DType>, Vec<Wire<DType>>) {
    let rule = expr_pair.as_rule();

    fn enforce(
        final_write: &Option<Wire<DType>>,
        terms: &mut Vec<Term<DType, IType>>,
        w: Wire<DType>,
        r: Vec<Wire<DType>>,
    ) -> (Wire<DType>, Vec<Wire<DType>>) {
        match final_write {
            Some(fw) if fw != &w => {
                terms.push(
                    Term::function(IType::Assign, [fw.clone()], vec![w.clone()]).unwrap(),
                );
                (fw.clone(), r)
            }
            _ => (w, r),
        }
    }

    match rule {
        Rule::expr_cond => {
            let mut inner = expr_pair.into_inner();
            let cond_pair = inner.next().unwrap();
            if inner.peek().is_none() {
                return lower_expr_to_terms(cond_pair, ctx, final_write, temp_index, terms);
            }
            inner.next(); // skip '?'
            let then_pair = inner.next().unwrap();
            inner.next(); // skip ':'
            let else_pair = inner.next().unwrap();

            let (cond_w, cond_read) =
                lower_expr_to_terms(cond_pair, ctx, None, temp_index, terms);
            let (then_w, then_read) =
                lower_expr_to_terms(then_pair, ctx, None, temp_index, terms);
            let (else_w, else_read) =
                lower_expr_to_terms(else_pair, ctx, None, temp_index, terms);

            let write = match &final_write {
                Some(w) => w.clone(),
                None => {
                    let w = Wire::new(*temp_index, DType::Int);
                    *temp_index += 1;
                    w
                }
            };

            terms.push(
                Term::function(
                    IType::Cond,
                    [write.clone()],
                    vec![cond_w.clone(), then_w.clone(), else_w.clone()],
                )
                .unwrap(),
            );

            let mut read_union = vec![];
            read_union.extend_from_slice(&cond_read);
            read_union.extend_from_slice(&then_read);
            read_union.extend_from_slice(&else_read);
            enforce(&final_write, terms, write, read_union)
        }

        Rule::expr_implies => {
            // right-associative: expr_iff (IMPLIES_OP expr_implies)?
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (left_w, mut left_read) =
                lower_expr_to_terms(first, ctx, None, temp_index, terms);
            if let Some(_op) = inner.next() {
                // IMPLIES_OP
                let rhs = inner.next().unwrap();
                let (right_w, right_read) =
                    lower_expr_to_terms(rhs, ctx, None, temp_index, terms);
                let write = match &final_write {
                    Some(w) => w.clone(),
                    None => {
                        let w = Wire::new(*temp_index, DType::Bool);
                        *temp_index += 1;
                        w
                    }
                };
                terms.push(
                    Term::function(
                        IType::Implies,
                        [write.clone()],
                        vec![left_w.clone(), right_w.clone()],
                    )
                    .unwrap(),
                );
                left_read.extend_from_slice(&right_read);
                return enforce(&final_write, terms, write, left_read);
            }
            enforce(&final_write, terms, left_w, left_read)
        }

        Rule::expr_iff => {
            // left-assoc binary: expr_or ((IFF | XNOR) expr_or)*
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (mut left_w, mut left_read) =
                lower_expr_to_terms(first, ctx, None, temp_index, terms);
            while let Some(_op) = inner.next() {
                let rhs = inner.next().unwrap();
                let (right_w, right_read) =
                    lower_expr_to_terms(rhs, ctx, None, temp_index, terms);
                let write = match &final_write {
                    Some(w) => w.clone(),
                    None => {
                        let w = Wire::new(*temp_index, DType::Bool);
                        *temp_index += 1;
                        w
                    }
                };
                terms.push(
                    Term::function(
                        IType::Xnor,
                        [write.clone()],
                        vec![left_w.clone(), right_w.clone()],
                    )
                    .unwrap(),
                );
                left_read.extend_from_slice(&right_read);
                left_w = write;
            }
            enforce(&final_write, terms, left_w, left_read)
        }

        Rule::expr_or => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (mut left_w, mut left_read) =
                lower_expr_to_terms(first, ctx, None, temp_index, terms);
            while let Some(op) = inner.next() {
                let rhs = inner.next().unwrap();
                let (right_w, right_read) =
                    lower_expr_to_terms(rhs, ctx, None, temp_index, terms);
                let write = match &final_write {
                    Some(w) => w.clone(),
                    None => {
                        let w = Wire::new(*temp_index, DType::Bool);
                        *temp_index += 1;
                        w
                    }
                };
                let tag = match op.as_rule() {
                    Rule::XOR => IType::Xor,
                    _ => IType::Or,
                };
                terms.push(
                    Term::function(tag, [write.clone()], vec![left_w.clone(), right_w.clone()])
                        .unwrap(),
                );
                left_read.extend_from_slice(&right_read);
                left_w = write;
            }
            enforce(&final_write, terms, left_w, left_read)
        }

        Rule::expr_and => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (mut left_w, mut left_read) =
                lower_expr_to_terms(first, ctx, None, temp_index, terms);
            while let Some(_op) = inner.next() {
                let rhs = inner.next().unwrap();
                let (right_w, right_read) =
                    lower_expr_to_terms(rhs, ctx, None, temp_index, terms);
                let write = match &final_write {
                    Some(w) => w.clone(),
                    None => {
                        let w = Wire::new(*temp_index, DType::Bool);
                        *temp_index += 1;
                        w
                    }
                };
                terms.push(
                    Term::function(
                        IType::And,
                        [write.clone()],
                        vec![left_w.clone(), right_w.clone()],
                    )
                    .unwrap(),
                );
                left_read.extend_from_slice(&right_read);
                left_w = write;
            }
            enforce(&final_write, terms, left_w, left_read)
        }

        Rule::expr_not => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            if first.as_rule() == Rule::NOT {
                let operand = inner.next().unwrap();
                let (sub_w, sub_read) =
                    lower_expr_to_terms(operand, ctx, None, temp_index, terms);
                let write = match &final_write {
                    Some(w) => w.clone(),
                    None => {
                        let w = Wire::new(*temp_index, DType::Bool);
                        *temp_index += 1;
                        w
                    }
                };
                terms.push(
                    Term::function(IType::Not, [write.clone()], vec![sub_w]).unwrap(),
                );
                enforce(&final_write, terms, write, sub_read)
            } else {
                lower_expr_to_terms(first, ctx, final_write, temp_index, terms)
            }
        }

        Rule::expr_cmp => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (mut left_w, mut left_read) =
                lower_expr_to_terms(first, ctx, None, temp_index, terms);
            while let Some(op) = inner.next() {
                let right_pair = inner.next().unwrap();
                let (right_w, right_read) =
                    lower_expr_to_terms(right_pair, ctx, None, temp_index, terms);
                let write = Wire::new(*temp_index, DType::Bool);
                *temp_index += 1;
                let tag = match op.as_rule() {
                    Rule::LT => IType::Lt,
                    Rule::LE => IType::Le,
                    Rule::GT => IType::Gt,
                    Rule::GE => IType::Ge,
                    Rule::EQ => IType::Eq,
                    Rule::NEQ => IType::Neq,
                    _ => unreachable!("unexpected cmp op"),
                };
                terms.push(
                    Term::function(tag, [write.clone()], vec![left_w.clone(), right_w.clone()])
                        .unwrap(),
                );
                left_read.extend_from_slice(&right_read);
                left_w = write;
            }
            enforce(&final_write, terms, left_w, left_read)
        }

        Rule::expr_arith => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (mut left_w, mut left_read) =
                lower_expr_to_terms(first, ctx, None, temp_index, terms);
            while let Some(op) = inner.next() {
                let right_pair = inner.next().unwrap();
                let (right_w, right_read) =
                    lower_expr_to_terms(right_pair, ctx, None, temp_index, terms);
                let write = Wire::new(*temp_index, DType::Int);
                *temp_index += 1;
                let tag = match op.as_rule() {
                    Rule::PLUS => IType::Add,
                    Rule::MINUS => IType::Sub,
                    _ => unreachable!("unexpected arith op"),
                };
                terms.push(
                    Term::function(tag, [write.clone()], vec![left_w.clone(), right_w.clone()])
                        .unwrap(),
                );
                left_read.extend_from_slice(&right_read);
                left_w = write;
            }
            enforce(&final_write, terms, left_w, left_read)
        }

        Rule::expr_mod => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (mut left_w, mut left_read) =
                lower_expr_to_terms(first, ctx, None, temp_index, terms);
            while let Some(_op) = inner.next() {
                let right_pair = inner.next().unwrap();
                let (right_w, right_read) =
                    lower_expr_to_terms(right_pair, ctx, None, temp_index, terms);
                let write = Wire::new(*temp_index, DType::Int);
                *temp_index += 1;
                terms.push(
                    Term::function(
                        IType::Mod,
                        [write.clone()],
                        vec![left_w.clone(), right_w.clone()],
                    )
                    .unwrap(),
                );
                left_read.extend_from_slice(&right_read);
                left_w = write;
            }
            enforce(&final_write, terms, left_w, left_read)
        }

        Rule::expr_term => {
            let mut inner = expr_pair.into_inner();
            let first = inner.next().unwrap();
            let (mut left_w, mut left_read) =
                lower_expr_to_terms(first, ctx, None, temp_index, terms);
            while let Some(op) = inner.next() {
                let right_pair = inner.next().unwrap();
                let (right_w, right_read) =
                    lower_expr_to_terms(right_pair, ctx, None, temp_index, terms);
                let write = Wire::new(*temp_index, DType::Int);
                *temp_index += 1;
                let tag = match op.as_rule() {
                    Rule::TIMES => IType::Mul,
                    Rule::DIVIDE => IType::Div,
                    _ => unreachable!("unexpected term op"),
                };
                terms.push(
                    Term::function(tag, [write.clone()], vec![left_w.clone(), right_w.clone()])
                        .unwrap(),
                );
                left_read.extend_from_slice(&right_read);
                left_w = write;
            }
            enforce(&final_write, terms, left_w, left_read)
        }

        Rule::neg_expr => {
            // MINUS ~ (neg_expr | expr_primary)
            let mut inner = expr_pair.into_inner();
            inner.next(); // skip MINUS
            let operand = inner.next().unwrap();
            let (sub_w, sub_read) =
                lower_expr_to_terms(operand, ctx, None, temp_index, terms);
            let write = match &final_write {
                Some(w) => w.clone(),
                None => {
                    let w = Wire::new(*temp_index, DType::Int);
                    *temp_index += 1;
                    w
                }
            };
            terms.push(
                Term::function(IType::Neg, [write.clone()], vec![sub_w]).unwrap(),
            );
            enforce(&final_write, terms, write, sub_read)
        }

        Rule::case_expr => {
            // case c1:v1; c2:v2; ... cN:vN; esac
            // Lowered to nested Cond: Cond(c1, v1, Cond(c2, v2, ... vN))
            let branches: Vec<Pair<Rule>> = expr_pair
                .into_inner()
                .filter(|p| p.as_rule() == Rule::case_branch)
                .collect();
            if branches.is_empty() {
                return enforce(&final_write, terms, Wire::new(0, DType::Bool), vec![]);
            }
            lower_case_branches(&branches, 0, ctx, final_write, temp_index, terms)
        }

        Rule::next_expr => {
            // next(expr) — reading a next-state value
            let inner_expr = expr_pair
                .into_inner()
                .find(|p| {
                    !matches!(p.as_rule(), Rule::NEXT)
                })
                .unwrap();
            // For simple ident, map to the next-wire
            if inner_expr.as_rule() == Rule::ident {
                let name = inner_expr.as_str().to_string();
                if let Some((_, idx, dtype)) = ctx.wires.iter().find(|(n, _, _)| n == &name) {
                    let next_wire = Wire::new(*idx + ctx.n, *dtype);
                    return enforce(&final_write, terms, next_wire.clone(), vec![next_wire]);
                }
            }
            // Fallback: lower the inner expression (limited support)
            lower_expr_to_terms(inner_expr, ctx, final_write, temp_index, terms)
        }

        Rule::expr_assign => {
            let mut inner = expr_pair.into_inner();
            let _target = inner.next().unwrap();
            let value_expr = inner.next().unwrap();
            lower_expr_to_terms(value_expr, ctx, final_write, temp_index, terms)
        }

        Rule::number => {
            let sval = expr_pair.as_str();
            let parsed = sval.parse::<i64>().expect("invalid numeric literal");
            let write = match &final_write {
                Some(w) => w.clone(),
                None => {
                    let w = Wire::new(*temp_index, DType::Int);
                    *temp_index += 1;
                    w
                }
            };
            terms.push(
                Term::function::<Wire<DType>, Wire<DType>, _, _>(
                    IType::ConstInt(parsed),
                    [write.clone()],
                    vec![],
                )
                .unwrap(),
            );
            enforce(&final_write, terms, write, vec![])
        }

        Rule::ident => {
            let name = expr_pair.as_str().to_string();
            // Check DEFINE expansion first
            if let Some(def_expr) = ctx.defines.get(&name) {
                if ctx.expanding.contains(&name) {
                    panic!("circular DEFINE: {}", name);
                }
                ctx.expanding.insert(name.clone());
                let result =
                    lower_expr_to_terms(def_expr.clone(), ctx, final_write, temp_index, terms);
                ctx.expanding.remove(&name);
                return result;
            }
            // Check enum values
            if let Some(&int_val) = ctx.enum_values.get(&name) {
                let write = match &final_write {
                    Some(w) => w.clone(),
                    None => {
                        let w = Wire::new(*temp_index, DType::Int);
                        *temp_index += 1;
                        w
                    }
                };
                terms.push(
                    Term::function::<Wire<DType>, Wire<DType>, _, _>(
                        IType::ConstInt(int_val),
                        [write.clone()],
                        vec![],
                    )
                    .unwrap(),
                );
                return enforce(&final_write, terms, write, vec![]);
            }
            // Wire lookup
            if let Some((_, idx, dtype)) = ctx.wires.iter().find(|(n, _, _)| n == &name) {
                let wire = Wire::new(*idx, *dtype);
                enforce(&final_write, terms, wire.clone(), vec![wire])
            } else {
                enforce(&final_write, terms, Wire::new(0, DType::Bool), vec![])
            }
        }

        Rule::TRUE => {
            let write = match &final_write {
                Some(w) => w.clone(),
                None => {
                    let w = Wire::new(*temp_index, DType::Bool);
                    *temp_index += 1;
                    w
                }
            };
            terms.push(
                Term::function::<Wire<DType>, Wire<DType>, _, _>(
                    IType::ConstBool(true),
                    [write.clone()],
                    vec![],
                )
                .unwrap(),
            );
            enforce(&final_write, terms, write, vec![])
        }

        Rule::FALSE => {
            let write = match &final_write {
                Some(w) => w.clone(),
                None => {
                    let w = Wire::new(*temp_index, DType::Bool);
                    *temp_index += 1;
                    w
                }
            };
            terms.push(
                Term::function::<Wire<DType>, Wire<DType>, _, _>(
                    IType::ConstBool(false),
                    [write.clone()],
                    vec![],
                )
                .unwrap(),
            );
            enforce(&final_write, terms, write, vec![])
        }

        Rule::word_lit_expr => {
            let mut children = expr_pair.into_inner();
            let word_lit = children.next().unwrap(); // word_literal
            let s = word_lit.as_str();
            // Parse word literal: 0ud16_0 or 0sd32_300
            let signed = s.as_bytes()[1] == b's';
            let rest = &s[3..]; // skip "0ud" or "0sd"
            let underscore_pos = rest.find('_').unwrap();
            let width: u32 = rest[..underscore_pos].parse().unwrap();
            let value: i64 = rest[underscore_pos + 1..].parse().unwrap();
            let dtype = if signed {
                DType::SWord(width)
            } else {
                DType::UWord(width)
            };

            let write = match &final_write {
                Some(w) => w.clone(),
                None => {
                    let w = Wire::new(*temp_index, dtype);
                    *temp_index += 1;
                    w
                }
            };
            terms.push(
                Term::function::<Wire<DType>, Wire<DType>, _, _>(
                    IType::ConstInt(value),
                    [write.clone()],
                    vec![],
                )
                .unwrap(),
            );

            // Check for bit_select
            if let Some(bs) = children.next()
                && bs.as_rule() == Rule::bit_select
            {
                let mut bs_inner = bs.into_inner();
                let high: u32 = bs_inner.next().unwrap().as_str().parse().unwrap();
                let low: u32 = bs_inner.next().unwrap().as_str().parse().unwrap();
                let bs_write = match &final_write {
                    Some(w) => w.clone(),
                    None => {
                        let w = Wire::new(*temp_index, DType::UWord(high - low + 1));
                        *temp_index += 1;
                        w
                    }
                };
                terms.push(
                    Term::function(
                        IType::BitSelect(high, low),
                        [bs_write.clone()],
                        vec![write.clone()],
                    )
                    .unwrap(),
                );
                return enforce(&final_write, terms, bs_write, vec![]);
            }
            enforce(&final_write, terms, write, vec![])
        }

        Rule::paren_expr => {
            let mut inner_expr_pair = None;
            let mut bit_sel = None;
            for child in expr_pair.into_inner() {
                if is_expr_rule(child.as_rule()) {
                    inner_expr_pair = Some(child);
                } else if child.as_rule() == Rule::bit_select {
                    bit_sel = Some(child);
                }
            }
            let (w, reads) = lower_expr_to_terms(
                inner_expr_pair.unwrap(),
                ctx,
                if bit_sel.is_some() { None } else { final_write.clone() },
                temp_index,
                terms,
            );
            if let Some(bs) = bit_sel {
                let mut bs_inner = bs.into_inner();
                let high: u32 = bs_inner.next().unwrap().as_str().parse().unwrap();
                let low: u32 = bs_inner.next().unwrap().as_str().parse().unwrap();
                let bs_write = match &final_write {
                    Some(fw) => fw.clone(),
                    None => {
                        let bw = Wire::new(*temp_index, DType::UWord(high - low + 1));
                        *temp_index += 1;
                        bw
                    }
                };
                terms.push(
                    Term::function(
                        IType::BitSelect(high, low),
                        [bs_write.clone()],
                        vec![w],
                    )
                    .unwrap(),
                );
                return enforce(&final_write, terms, bs_write, reads);
            }
            enforce(&final_write, terms, w, reads)
        }

        Rule::builtin_call => {
            let mut children_iter = expr_pair.into_inner();
            let fn_pair = children_iter.next().unwrap(); // BUILTIN_FN
            let fn_name = fn_pair.as_str();
            let first_arg = children_iter.find(|p| is_expr_rule(p.as_rule())).unwrap();
            let (arg_w, arg_reads) =
                lower_expr_to_terms(first_arg, ctx, None, temp_index, terms);

            let (itype, out_dtype) = match fn_name {
                "bool" => (IType::ToBool, DType::Bool),
                "word1" => (IType::ToWord1, DType::UWord(1)),
                "unsigned" => (IType::ToUnsigned, DType::UWord(32)),
                "signed" => (IType::ToSigned, DType::SWord(32)),
                "extend" => {
                    let width_pair = children_iter
                        .find(|p| p.as_rule() == Rule::number || is_expr_rule(p.as_rule()))
                        .unwrap();
                    // Peel down to the number leaf
                    let mut leaf = width_pair;
                    while let Some(inner) = leaf.clone().into_inner().next() {
                        if inner.as_rule() == Rule::number {
                            leaf = inner;
                            break;
                        }
                        leaf = inner;
                    }
                    let width: u32 = leaf.as_str().parse().unwrap();
                    (IType::Extend(width), DType::UWord(32))
                }
                _ => unreachable!("unexpected builtin function: {}", fn_name),
            };

            let write = match &final_write {
                Some(w) => w.clone(),
                None => {
                    let w = Wire::new(*temp_index, out_dtype);
                    *temp_index += 1;
                    w
                }
            };
            terms.push(
                Term::function(itype, [write.clone()], vec![arg_w]).unwrap(),
            );
            enforce(&final_write, terms, write, arg_reads)
        }

        Rule::abs_call => {
            let mut inner_expr: Option<Pair<Rule>> = None;
            for child in expr_pair.clone().into_inner() {
                match child.as_rule() {
                    r if is_expr_rule(r) => {
                        inner_expr = Some(child);
                        break;
                    }
                    _ => {}
                }
            }
            let inner = match inner_expr {
                Some(p) => p,
                None => {
                    return enforce(&final_write, terms, Wire::new(0, DType::Int), vec![])
                }
            };
            let (inner_w, inner_read) =
                lower_expr_to_terms(inner, ctx, None, temp_index, terms);

            let mut abs_read_for_term = vec![];
            let i = inner_w.id();
            let dt = inner_w.dtype();
            if i < ctx.n && i >= ctx.var_count {
                abs_read_for_term.push(Wire::new(i + ctx.n, *dt));
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

            terms.push(
                Term::function(IType::Abs, [write.clone()], abs_read_for_term).unwrap(),
            );
            enforce(&final_write, terms, write, inner_read)
        }

        _ => {
            if let Some(inner) = expr_pair.clone().into_inner().next() {
                lower_expr_to_terms(inner, ctx, final_write, temp_index, terms)
            } else {
                enforce(&final_write, terms, Wire::new(0, DType::Bool), vec![])
            }
        }
    }
}

/// Recursively lower case branches into nested Cond terms.
fn lower_case_branches(
    branches: &[Pair<Rule>],
    idx: usize,
    ctx: &mut LowerCtx,
    final_write: Option<Wire<DType>>,
    temp_index: &mut usize,
    terms: &mut Vec<Term<DType, IType>>,
) -> (Wire<DType>, Vec<Wire<DType>>) {
    fn enforce(
        final_write: &Option<Wire<DType>>,
        terms: &mut Vec<Term<DType, IType>>,
        w: Wire<DType>,
        r: Vec<Wire<DType>>,
    ) -> (Wire<DType>, Vec<Wire<DType>>) {
        match final_write {
            Some(fw) if fw != &w => {
                terms.push(
                    Term::function(IType::Assign, [fw.clone()], vec![w.clone()]).unwrap(),
                );
                (fw.clone(), r)
            }
            _ => (w, r),
        }
    }

    if idx >= branches.len() {
        // shouldn't happen, but fallback
        return (Wire::new(0, DType::Int), vec![]);
    }

    let branch = branches[idx].clone();
    let mut branch_inner = branch.into_inner();
    let cond_expr = branch_inner.next().unwrap();
    let value_expr = branch_inner.next().unwrap();

    // Last branch: just return the value (acts as the else fallback)
    if idx == branches.len() - 1 {
        return lower_expr_to_terms(value_expr, ctx, final_write, temp_index, terms);
    }

    let (cond_w, cond_read) =
        lower_expr_to_terms(cond_expr, ctx, None, temp_index, terms);
    let (then_w, then_read) =
        lower_expr_to_terms(value_expr, ctx, None, temp_index, terms);
    let (else_w, else_read) =
        lower_case_branches(branches, idx + 1, ctx, None, temp_index, terms);

    let write = match &final_write {
        Some(w) => w.clone(),
        None => {
            let w = Wire::new(*temp_index, DType::Int);
            *temp_index += 1;
            w
        }
    };

    terms.push(
        Term::function(
            IType::Cond,
            [write.clone()],
            vec![cond_w.clone(), then_w.clone(), else_w.clone()],
        )
        .unwrap(),
    );

    let mut read_union = vec![];
    read_union.extend_from_slice(&cond_read);
    read_union.extend_from_slice(&then_read);
    read_union.extend_from_slice(&else_read);
    enforce(&final_write, terms, write, read_union)
}

#[derive(Parser)]
#[grammar = "smv.pest"]
pub struct SMVParser;

pub fn parse_smv(input: &str) -> Result<ParseResult, &'static str> {
    let parsed = SMVParser::parse(Rule::file, input)
        .map_err(|e| {
            eprintln!("Pest parse error: {}", e);
            "parse failed"
        })?
        .next()
        .ok_or("empty parse tree")?;
    build_module(parsed)
}

/// Resolve a type_spec pair to a DType. Supports boolean, integer, range types,
/// and enum types (all mapped to Int for enums/ranges).
fn resolve_type(
    type_pair: &Pair<Rule>,
    enum_values: &mut HashMap<String, i64>,
) -> Result<DType, &'static str> {
    // type_spec is non-silent, so its children are the actual type alternatives
    let inner = type_pair.clone().into_inner().next().unwrap_or_else(|| type_pair.clone());
    match inner.as_rule() {
        Rule::BOOLEAN => Ok(DType::Bool),
        Rule::INTEGER => Ok(DType::Int),
        Rule::range_type => Ok(DType::Int), // ignore bounds
        Rule::word_type => {
            let mut signed = false;
            let mut width = 0u32;
            for child in inner.into_inner() {
                match child.as_rule() {
                    Rule::SIGNED_KW => signed = true,
                    Rule::number => width = child.as_str().parse().unwrap(),
                    _ => {}
                }
            }
            if signed {
                Ok(DType::SWord(width))
            } else {
                Ok(DType::UWord(width))
            }
        }
        Rule::enum_type => {
            // Collect enum values and assign sequential integer codes
            let mut code: i64 = 0;
            for ev in inner.into_inner() {
                if ev.as_rule() == Rule::enum_value {
                    let val_inner = ev.into_inner().next().unwrap();
                    if val_inner.as_rule() == Rule::ident {
                        let name = val_inner.as_str().to_string();
                        enum_values.entry(name).or_insert(code);
                    }
                    // number enum values are their own int code; skip adding to map
                    code += 1;
                }
            }
            Ok(DType::Int)
        }
        _ => {
            // Fallback: try text matching for direct keyword matches
            match type_pair.as_str().trim() {
                "boolean" => Ok(DType::Bool),
                "integer" => Ok(DType::Int),
                _ => Err("unsupported type"),
            }
        }
    }
}

fn build_module(file_pair: Pair<Rule>) -> Result<ParseResult, &'static str> {
    let mut var_decls: Vec<(String, DType)> = vec![];
    let mut ivar_decls: Vec<(String, DType)> = vec![];
    let mut frozen_decls: Vec<(String, DType)> = vec![];
    let mut wires: Vec<(String, usize, DType)> = vec![];
    let mut init_assigns: Vec<(String, Pair<Rule>)> = vec![];
    let mut next_assigns: Vec<(String, Pair<Rule>)> = vec![];
    let mut invar_exprs: Vec<Pair<Rule>> = vec![];
    let mut init_exprs: Vec<Pair<Rule>> = vec![];
    let mut trans_exprs: Vec<Pair<Rule>> = vec![];
    let mut define_map: HashMap<String, Pair<Rule>> = HashMap::new();
    let mut enum_values: HashMap<String, i64> = HashMap::new();

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
                            let type_pair = decl_iter.next().unwrap();
                            let dtype = resolve_type(&type_pair, &mut enum_values)?;
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
                            let type_pair = decl_iter.next().unwrap();
                            let dtype = resolve_type(&type_pair, &mut enum_values)?;
                            var_decls.push((name, dtype));
                        }
                    }

                    Rule::frozenvar_section => {
                        for decl in body_item
                            .into_inner()
                            .filter(|p| p.as_rule() == Rule::frozenvar_decl)
                        {
                            let mut decl_iter = decl.into_inner();
                            let name = decl_iter.next().unwrap().as_str().to_string();
                            let type_pair = decl_iter.next().unwrap();
                            let dtype = resolve_type(&type_pair, &mut enum_values)?;
                            frozen_decls.push((name, dtype));
                        }
                    }

                    Rule::define_section => {
                        for decl in body_item
                            .into_inner()
                            .filter(|p| p.as_rule() == Rule::define_decl)
                        {
                            let mut decl_iter = decl.into_inner();
                            let name = decl_iter.next().unwrap().as_str().to_string();
                            let expr = decl_iter.next().unwrap();
                            define_map.insert(name, expr);
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
                                    let var_name = target_pair.as_str().to_string();
                                    init_assigns.push((var_name, expr_pair));
                                }
                                _ => {}
                            }
                        }
                    }

                    Rule::invar_section => {
                        for child in body_item.into_inner() {
                            if is_expr_rule(child.as_rule()) {
                                invar_exprs.push(child);
                            }
                        }
                    }

                    Rule::init_section => {
                        for child in body_item.into_inner() {
                            if is_expr_rule(child.as_rule()) {
                                init_exprs.push(child);
                            }
                        }
                    }

                    Rule::trans_section => {
                        for child in body_item.into_inner() {
                            if is_expr_rule(child.as_rule()) {
                                trans_exprs.push(child);
                            }
                        }
                    }

                    // Parsed but ignored
                    Rule::fairness_section
                    | Rule::justice_section
                    | Rule::compassion_section
                    | Rule::constants_section => {}

                    _ => {}
                }
            }
        }
    }

    // Prepend frozen vars before regular state vars so their init terms
    // are emitted first (other vars' init may read frozen var primed wires).
    {
        let mut combined = frozen_decls
            .iter()
            .map(|(n, d)| (n.clone(), *d))
            .collect::<Vec<_>>();
        combined.append(&mut var_decls);
        var_decls = combined;
    }

    // Assemble wires: VAR first, then IVAR
    for (name, dtype) in var_decls.iter() {
        let index = wires.len();
        wires.push((name.clone(), index, *dtype));
    }
    for (name, dtype) in ivar_decls.iter() {
        let index = wires.len();
        wires.push((name.clone(), index, *dtype));
    }

    let n = wires.len();
    let var_count = var_decls.len();

    // Auto-generate identity updates for frozen vars
    let frozen_names: HashSet<String> = frozen_decls.iter().map(|(n, _)| n.clone()).collect();
    for (name, _) in &frozen_decls {
        if !next_assigns.iter().any(|(n, _)| n == name) {
            // We need a dummy Pair for the identity assignment.
            // Instead, we'll handle this specially in the update loop below.
        }
    }

    // Step 2: latched and next wires
    let mut latched = vec![];
    let mut next_wire = vec![];
    for (_, i, dtype) in &wires {
        latched.push(Wire::new(*i, *dtype));
        next_wire.push(Wire::new(i + n, *dtype));
    }

    // Step 3: build per-variable init and update Terms
    let mut init_terms: Vec<Term<DType, IType>> = vec![];
    let mut update_terms: Vec<Term<DType, IType>> = vec![];

    let mut temp_index: usize = 2 * n;

    let mut ctx = LowerCtx {
        wires: &wires,
        var_count,
        n,
        defines: &define_map,
        enum_values: &enum_values,
        expanding: HashSet::new(),
    };

    for i in 0..var_count {
        let (name, idx, dtype) = &wires[i];

        // INIT
        if let Some((_, expr_pair)) = init_assigns.iter().find(|(n, _)| n == name) {
            let write = Wire::new(idx + n, *dtype);
            // detect simple single-leaf RHS
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
                    let parsed = sval.parse::<i64>().map_err(|_| "invalid numeric literal")?;
                    init_terms.push(
                        Term::function::<Wire<DType>, Wire<DType>, _, _>(
                            IType::ConstInt(parsed),
                            [write.clone()],
                            vec![],
                        )
                        .unwrap(),
                    );
                }
                Rule::neg_expr => {
                    // unary minus, e.g. init(x) := -1
                    lower_expr_to_terms(
                        expr_pair.clone(),
                        &mut ctx,
                        Some(write.clone()),
                        &mut temp_index,
                        &mut init_terms,
                    );
                }
                Rule::ident => {
                    let nm = leaf.as_str().to_string();
                    // Check enum values first
                    if let Some(&int_val) = enum_values.get(&nm) {
                        init_terms.push(
                            Term::function::<Wire<DType>, Wire<DType>, _, _>(
                                IType::ConstInt(int_val),
                                [write.clone()],
                                vec![],
                            )
                            .unwrap(),
                        );
                    } else if let Some((_, ridx, rdtype)) =
                        wires.iter().find(|(n, _, _)| n == &nm)
                    {
                        // In init context, always read the primed (next) wire.
                        // Module::sequential requires init terms to only read
                        // next wires, not latched wires.
                        let primed = Wire::new(*ridx + n, *rdtype);
                        init_terms.push(
                            Term::function(IType::Assign, [write.clone()], vec![primed])
                                .unwrap(),
                        );
                    } else {
                        // Could be a DEFINE or enum — use full lowering
                        lower_expr_to_terms(
                            expr_pair.clone(),
                            &mut ctx,
                            Some(write.clone()),
                            &mut temp_index,
                            &mut init_terms,
                        );
                    }
                }
                Rule::abs_call => {
                    if let Some(inner) = leaf.clone().into_inner().next()
                        && inner.as_rule() == Rule::ident
                    {
                        let nm = inner.as_str().to_string();
                        if let Some((_, ridx, rdtype)) =
                            wires.iter().find(|(n, _, _)| n == &nm)
                        {
                            if *ridx < var_count {
                                let read = Wire::new(*ridx, *rdtype);
                                init_terms.push(
                                    Term::function(
                                        IType::Assign,
                                        [write.clone()],
                                        vec![read],
                                    )
                                    .unwrap(),
                                );
                            } else {
                                let primed = Wire::new(*ridx + n, *rdtype);
                                init_terms.push(
                                    Term::function(
                                        IType::Assign,
                                        [write.clone()],
                                        vec![primed],
                                    )
                                    .unwrap(),
                                );
                            }
                            continue;
                        }
                    }
                    // Fallback: lower the full abs expression
                    let mut inner_expr: Option<Pair<Rule>> = None;
                    for child in leaf.clone().into_inner() {
                        if is_expr_rule(child.as_rule()) {
                            inner_expr = Some(child);
                            break;
                        }
                    }

                    let (inner_w, _inner_read) = if let Some(p) = inner_expr {
                        lower_expr_to_terms(
                            p,
                            &mut ctx,
                            None,
                            &mut temp_index,
                            &mut init_terms,
                        )
                    } else {
                        lower_expr_to_terms(
                            leaf.clone(),
                            &mut ctx,
                            None,
                            &mut temp_index,
                            &mut init_terms,
                        )
                    };

                    let ii = inner_w.id();
                    let dt = inner_w.dtype();
                    let abs_read = if ii < n && ii >= var_count {
                        Wire::new(ii + n, *dt)
                    } else {
                        Wire::new(ii, *dt)
                    };

                    let temp_w = Wire::new(temp_index, DType::Int);
                    temp_index += 1;
                    init_terms.push(
                        Term::function(IType::Abs, [temp_w.clone()], vec![abs_read]).unwrap(),
                    );
                    init_terms.push(
                        Term::function(IType::Assign, [write.clone()], vec![temp_w]).unwrap(),
                    );
                }
                _ => {
                    lower_expr_to_terms(
                        expr_pair.clone(),
                        &mut ctx,
                        Some(write.clone()),
                        &mut temp_index,
                        &mut init_terms,
                    );
                }
            }
        } else {
            // default init: constant zero / false
            let default_val = match dtype {
                DType::Bool => IType::ConstBool(false),
                DType::Int | DType::UWord(_) | DType::SWord(_) => IType::ConstInt(0),
            };
            let write = Wire::new(idx + n, *dtype);
            init_terms.push(
                Term::function::<Wire<DType>, Wire<DType>, _, _>(default_val, [write], vec![])
                    .unwrap(),
            );
        }

        // UPDATE
        if frozen_names.contains(name) && !next_assigns.iter().any(|(n, _)| n == name) {
            // Frozen var: identity update next(x) := x
            let write = Wire::new(idx + n, *dtype);
            let read = Wire::new(*idx, *dtype);
            update_terms
                .push(Term::function(IType::Assign, [write], vec![read]).unwrap());
        } else if let Some((_, expr_pair)) = next_assigns.iter().find(|(n, _)| n == name) {
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
                    let parsed =
                        sval.parse::<i64>().map_err(|_| "invalid numeric literal")?;
                    update_terms.push(
                        Term::function::<Wire<DType>, Wire<DType>, _, _>(
                            IType::ConstInt(parsed),
                            [write.clone()],
                            vec![],
                        )
                        .unwrap(),
                    );
                }
                Rule::ident => {
                    let nm = leaf.as_str().to_string();
                    if let Some((_, ridx, rdtype)) =
                        wires.iter().find(|(n, _, _)| n == &nm)
                    {
                        let read = Wire::new(*ridx, *rdtype);
                        update_terms.push(
                            Term::function(IType::Assign, [write.clone()], vec![read])
                                .unwrap(),
                        );
                    } else {
                        lower_expr_to_terms(
                            expr_pair.clone(),
                            &mut ctx,
                            Some(write.clone()),
                            &mut temp_index,
                            &mut update_terms,
                        );
                    }
                }
                Rule::abs_call => {
                    if let Some(inner) = leaf.clone().into_inner().next()
                        && inner.as_rule() == Rule::ident
                    {
                        let nm = inner.as_str().to_string();
                        if let Some((_, ridx, rdtype)) =
                            wires.iter().find(|(n, _, _)| n == &nm)
                        {
                            let read = Wire::new(*ridx, *rdtype);
                            update_terms.push(
                                Term::function(
                                    IType::Assign,
                                    [write.clone()],
                                    vec![read],
                                )
                                .unwrap(),
                            );
                            continue;
                        }
                    }
                    lower_expr_to_terms(
                        expr_pair.clone(),
                        &mut ctx,
                        Some(write.clone()),
                        &mut temp_index,
                        &mut update_terms,
                    );
                }
                _ => {
                    lower_expr_to_terms(
                        expr_pair.clone(),
                        &mut ctx,
                        Some(write.clone()),
                        &mut temp_index,
                        &mut update_terms,
                    );
                }
            }
        } else {
            // default update: next(var) := var
            let write = Wire::new(idx + n, *dtype);
            let read = Wire::new(*idx, *dtype);
            update_terms
                .push(Term::function(IType::Assign, [write], vec![read]).unwrap());
        }
    }

    // Step 4: lower constraint sections into terms
    let mut init_constraint_terms: Vec<Term<DType, IType>> = vec![];
    let mut trans_constraint_terms: Vec<Term<DType, IType>> = vec![];
    let mut invar_constraint_terms: Vec<Term<DType, IType>> = vec![];

    for expr in init_exprs {
        lower_expr_to_terms(
            expr,
            &mut ctx,
            None,
            &mut temp_index,
            &mut init_constraint_terms,
        );
    }
    for expr in trans_exprs {
        lower_expr_to_terms(
            expr,
            &mut ctx,
            None,
            &mut temp_index,
            &mut trans_constraint_terms,
        );
    }
    for expr in invar_exprs {
        lower_expr_to_terms(
            expr,
            &mut ctx,
            None,
            &mut temp_index,
            &mut invar_constraint_terms,
        );
    }

    // Step 5: construct module
    let obs_pairs: Vec<[Wire<DType>; 2]> = latched
        .iter()
        .zip(next_wire.iter())
        .map(|(l, n)| [l.clone(), n.clone()])
        .collect();
    let module = Module::sequential(
        obs_pairs,
        std::iter::empty::<[Wire<DType>; 2]>(),
        init_terms,
        update_terms,
    )?;

    Ok(ParseResult {
        module,
        init_constraints: init_constraint_terms,
        trans_constraints: trans_constraint_terms,
        invar_constraints: invar_constraint_terms,
    })
}

/// Check if a Rule corresponds to an expression-level rule
/// (used to filter children when extracting expressions from sections).
fn is_expr_rule(rule: Rule) -> bool {
    matches!(
        rule,
        Rule::expr_cond
            | Rule::expr_implies
            | Rule::expr_iff
            | Rule::expr_or
            | Rule::expr_and
            | Rule::expr_not
            | Rule::expr_cmp
            | Rule::expr_arith
            | Rule::expr_mod
            | Rule::expr_term
            | Rule::neg_expr
            | Rule::expr_assign
            | Rule::case_expr
            | Rule::next_expr
            | Rule::abs_call
            | Rule::builtin_call
            | Rule::word_lit_expr
            | Rule::word_literal
            | Rule::paren_expr
            | Rule::ident
            | Rule::number
            | Rule::TRUE
            | Rule::FALSE
    )
}
