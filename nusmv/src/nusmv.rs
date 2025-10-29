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

/// Build a `Module` from the top-level `file` parse `Pair`.
///
/// This function traverses the parse tree produced by the Pest grammar
/// and collects:
/// - variable declarations (collected into `wires` as (name, index, dtype))
/// - init assignments (collected into `init_assigns`)
/// - next assignments (collected into `next_assigns`)
///
/// After collecting these, it constructs the latched and next `Wire` pairs
/// and converts `next` assignments into `Atom`s (using `Term`s built from
/// expressions).
fn build_module(file_pair: Pair<Rule>) -> Result<Module<DType, IType>, &'static str> {
    let mut wires: Vec<(String, usize, DType)> = vec![];
    let mut init_assigns: Vec<(String, Pair<Rule>)> = vec![];
    let mut next_assigns: Vec<(String, Pair<Rule>)> = vec![];

    for section in file_pair.into_inner() {
        match section.as_rule() {
            Rule::module_decl => {
                for inner in section.into_inner() {
                    match inner.as_rule() {
                        Rule::module_body => {
                            for body_item in inner.into_inner() {
                                match body_item.as_rule() {
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
                                            let index = wires.len();
                                            wires.push((name, index, dtype));
                                        }
                                    }

                                    Rule::assign_section => {
                                        for assign in body_item.into_inner().filter(|p| p.as_rule() == Rule::assign_stmt) {
                                            let mut parts = assign.into_inner();
                                            let target_pair = parts.next().unwrap();
                                            let expr_pair = parts.next().unwrap();
                                            
                                            match target_pair.as_rule() {
                                                Rule::init_ref => {
                                                    let var_name = target_pair.into_inner().next().unwrap().as_str().to_string();
                                                    init_assigns.push((var_name, expr_pair));
                                                }
                                                Rule::next_ref => {
                                                    let var_name = target_pair.into_inner().next().unwrap().as_str().to_string();
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

                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    // Build wire pairs: [latched (0..n), next (n..2n)]
    let n = wires.len();
    
    let mut latched = Wire::empty();
    for (i, (_, _, dtype)) in wires.iter().enumerate() {
        latched = latched.union(&Wire::scalar(i, dtype.clone())).unwrap();
    }
    
    let mut next = Wire::empty();
    for (i, (_, _, dtype)) in wires.iter().enumerate() {
        next = next.union(&Wire::scalar(i + n, dtype.clone())).unwrap();
    }

    let mut atoms: Vec<Atom<DType, IType>> = vec![];
    
    for (var_name, expr_pair) in next_assigns {
        if let Some((_, idx, dtype)) = wires.iter().find(|(n, _, _)| n == &var_name) {
            // Extract the IType and read wires from the expression
            let (update_itype, _, read) = build_expr_helper(expr_pair.clone(), &wires, 0);
            let update_term = Term::new(update_itype, Wire::empty(), read.clone());
            
            // Build init term
            let init_term = init_assigns.iter()
                .find(|(n, _)| n == &var_name)
                .map(|(_, p)| {
                    let (itype, write, read) = build_expr_helper(p.clone(), &wires, 0);
                    Term::new(itype, write, read)
                })
                .unwrap_or_else(|| {
                    let default_val = match dtype {
                        DType::Bool => IType::ConstBool(false),
                        DType::Int => IType::ConstInt(0),
                    };
                    Term::new(default_val, Wire::empty(), Wire::empty())
                });
            
            // ctrl: the next wire being written to
            let ctrl = Wire::scalar(idx + n, dtype.clone());
            
            // wait: empty (NuSMV doesn't have explicit await)
            let wait = Wire::empty();
            
            // IMPORTANT: read must be a subset of the latched wires
            // If read is empty, that's fine - it means the expression is constant
            // But we need to ensure it's a valid subset
            let read_subset = if read.size() > 0 {
                read
            } else {
                // For constant expressions, read is empty - that's valid
                Wire::empty()
            };
            
            let atom = Atom::new_unchecked(
                ctrl,
                wait,
                read_subset,
                vec![init_term],
                vec![update_term],
            );
            
            atoms.push(atom);
        }
    }

    Module::new([latched, next], atoms).map_err(|_| "Failed to build module")
}

/// Build a `Term<DType, IType>` from a parse `Pair`.
///
/// Inputs:
/// - `expr_pair`: a Pest `Pair<Rule>` representing an expression node.
/// - `wires`: the declared variables as (name, index, dtype) used to
///   resolve identifiers to wire indices.
/// - `offset`: index offset used when constructing write wires (e.g. to
///   distinguish latched vs next index spaces).
///
/// Output:
/// - a `Term` bundling the instruction (`IType`) and its write/read wires.
///
/// Rationale:
/// - This function is a thin wrapper that calls `build_expr_helper` (the
///   recursive worker) which returns the decomposed components
///   `(IType, write_wire, read_wire)`. `build_expr` then wraps those into a
///   `Term` when callers need that concrete type (e.g., when creating
///   `Atom` init/update terms).
fn build_expr(expr_pair: Pair<Rule>, wires: &[(String, usize, DType)], offset: usize) -> Term<DType, IType> {
    let (itype, write, read) = build_expr_helper(expr_pair, wires, offset);
    Term::new(itype, write, read)
}

/// Decompose a parsed expression into (IType, write, read).
///
/// Returns:
/// - `IType`: instruction-like representation of the expression.
/// - `write: Wire<DType>`: the wire written by this term (often empty for
///   pure reads; boolean operators use a scalar write at `offset`).
/// - `read: Wire<DType>`: union of wires read by this expression.
///
/// Notes:
/// - `offset` is used to choose write indices for composed boolean terms.
/// - This helper exposes the raw components so parent expressions can
///   combine read sets and build composed `IType` nodes without needing
///   accessors on `Term` (which are not available in the `base` crate).
fn build_expr_helper(expr_pair: Pair<Rule>, wires: &[(String, usize, DType)], offset: usize) -> (IType, Wire<DType>, Wire<DType>) {
    match expr_pair.as_rule() {
        Rule::expr_not => {
            let inner = expr_pair.into_inner().next().unwrap();
            let (sub_itype, _, sub_read) = build_expr_helper(inner, wires, offset);
            (
                IType::Not(Box::new(sub_itype)),
                Wire::scalar(offset, DType::Bool),
                sub_read,
            )
        }

        Rule::expr_and => {
            let mut inner = expr_pair.into_inner();
            let (left_itype, _, left_read) = build_expr_helper(inner.next().unwrap(), wires, offset);
            let (right_itype, _, right_read) = build_expr_helper(inner.next().unwrap(), wires, offset);
            let combined_read = left_read.union(&right_read).unwrap_or(left_read.clone());
            (
                IType::And(Box::new(left_itype), Box::new(right_itype)),
                Wire::scalar(offset, DType::Bool),
                combined_read,
            )
        }

        Rule::expr_or => {
            let mut inner = expr_pair.into_inner();
            let (left_itype, _, left_read) = build_expr_helper(inner.next().unwrap(), wires, offset);
            let (right_itype, _, right_read) = build_expr_helper(inner.next().unwrap(), wires, offset);
            let combined_read = left_read.union(&right_read).unwrap_or(left_read.clone());
            (
                IType::Or(Box::new(left_itype), Box::new(right_itype)),
                Wire::scalar(offset, DType::Bool),
                combined_read,
            )
        }
        Rule::number => {
            let sval = expr_pair.as_str();
            let parsed = match sval.parse::<i64>() {
                Ok(v) => v,
                Err(e) => panic!("invalid numeric literal '{}': {}", sval, e),
            };
            (
                IType::ConstInt(parsed),
                Wire::empty(),
                Wire::empty(),
            )
        }
        Rule::ident => {
            let name = expr_pair.as_str().to_string();
            if let Some((_, idx, dtype)) = wires.iter().find(|(n, _, _)| n == &name) {
                (
                    IType::VarRef(name),
                    Wire::empty(),
                    Wire::scalar(idx + offset, dtype.clone()),
                )
            } else {
                (
                    IType::VarRef(name),
                    Wire::empty(),
                    Wire::empty(),
                )
            }
        }
        Rule::TRUE => {
            (
                IType::ConstBool(true),
                Wire::empty(),
                Wire::empty(),
            )
        }
        Rule::FALSE => {
            (
                IType::ConstBool(false),
                Wire::empty(),
                Wire::empty(),
            )
        }
        _ => {
            if let Some(inner) = expr_pair.clone().into_inner().next() {
                build_expr_helper(inner, wires, offset)
            } else {
                panic!("unhandled expr in helper: {:?}", expr_pair.as_rule())
            }
        }
    }
}
