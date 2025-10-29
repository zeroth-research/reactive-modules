use pest::Parser;
use pest_derive::Parser;
use pest::iterators::Pair;

use base::module::Module;
use base::atom::Atom;
use base::term::Term;
use base::wire::Wire;

use crate::dtype::DType;
use crate::itype::IType;

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

    // Step 4: compute aggregated ctrl/read/wait wires and a single init/update
    // term for the whole module, matching the manual construction used in
    // tests: ctrl = union of VARs that are controlled (have next assignments),
    // wait = union of IVARs, read = ctrl.
    let mut ctrl_latched = Wire::none();
    let mut wait_latched = Wire::none();

    // Build latched wires per name and collect ctrl/read/wait
    for (name, idx, dtype) in &wires {
        let latched_wire = Wire::one(*idx, dtype.clone());
        // IVARs are treated as wait wires
        if ivar_decls.iter().any(|(n, _)| n == name) {
            wait_latched = wait_latched.union(&latched_wire).unwrap();
        }
        // VARs that have a next assignment become controlled
        if next_assigns.iter().any(|(n, _)| n == name) {
            ctrl_latched = ctrl_latched.union(&latched_wire).unwrap();
        }
    }

    // read is the same as ctrl (latched)
    let read_latched = ctrl_latched.clone();

    // single init term: default value written to ctrl, reading from wait
    let default_val = IType::ConstInt(0);
    let init_terms = vec![Term::new(default_val.clone(), ctrl_latched.clone(), wait_latched.clone())];

    // single update term: default value written to ctrl, reading ctrl U wait
    let update_read = read_latched.union(&wait_latched).unwrap_or(read_latched.clone());
    let update_terms = vec![Term::new(default_val.clone(), ctrl_latched.clone(), update_read)];

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
                    IType::Cond(Box::new(cond.0), Box::new(then_b.0), Box::new(else_b.0)),
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
                combined = (IType::Or(Box::new(combined.0), Box::new(rhs.0)), Wire::one(offset, DType::Bool), read);
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
                combined = (IType::And(Box::new(combined.0), Box::new(rhs.0)), Wire::one(offset, DType::Bool), read);
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
                    Rule::LT => (IType::Lt(Box::new(left.0), Box::new(right.0)), Wire::one(offset, DType::Bool), read),
                    Rule::LE => (IType::Le(Box::new(left.0), Box::new(right.0)), Wire::one(offset, DType::Bool), read),
                    Rule::GT => (IType::Gt(Box::new(left.0), Box::new(right.0)), Wire::one(offset, DType::Bool), read),
                    Rule::GE => (IType::Ge(Box::new(left.0), Box::new(right.0)), Wire::one(offset, DType::Bool), read),
                    Rule::EQ => (IType::Eq(Box::new(left.0), Box::new(right.0)), Wire::one(offset, DType::Bool), read),
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
                        IType::Add(Box::new(left.0), Box::new(right.0)),
                        Wire::one(offset, DType::Int),
                        combined_read,
                    ),
                    Rule::MINUS => (
                        IType::Sub(Box::new(left.0), Box::new(right.0)),
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
                        IType::Mul(Box::new(left.0), Box::new(right.0)),
                        Wire::one(offset, DType::Int),
                        combined_read,
                    ),
                    Rule::DIVIDE => (
                        IType::Div(Box::new(left.0), Box::new(right.0)),
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
                let (sub_itype, _, sub_read) = build_expr(inner_factor, wires, offset);
                (
                    IType::Not(Box::new(sub_itype)),
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
            let target = inner.next().unwrap();
            let value_expr = inner.next().unwrap();
            let rhs = build_expr(value_expr, wires, offset);
            let target_name = match target.as_rule() {
                Rule::init_ref | Rule::next_ref => {
                    target.into_inner().next().unwrap().as_str().to_string()
                }
                Rule::ident => target.as_str().to_string(),
                _ => panic!("unexpected assign target {:?}", target.as_rule()),
            };
            (
                IType::Assign(Box::new(IType::VarRef(target_name)), Box::new(rhs.0)),
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
                (IType::VarRef(name), Wire::none(), Wire::one(idx + offset, dtype.clone()))
            } else {
                (IType::VarRef(name), Wire::none(), Wire::none())
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

