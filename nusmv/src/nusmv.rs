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
#[grammar = "nusmv.pest"] // path is relative to src/
pub struct NuSMVParser;

pub fn parse_nusmv(input: &str) {
    let parsed = NuSMVParser::parse(Rule::file, input)
        .expect("parse failed")
        .next()
        .unwrap();

    println!("{:#?}", parsed);

    // // Placeholder: build a dummy module for now
    // let atoms = vec![]; // You can replace this later with real atoms

    // Module {
    //     external: Wire::vector(0, DType::Bool, 1),
    //     interface: Wire::vector(1, DType::Bool, 1),
    //     private: Wire::vector(2, DType::Bool, 1),
    //     atoms,
    // }
}

fn build_module(file_pair: Pair<Rule>) -> Module<DType, IType> {

    let mut wires: Vec<(String, Wire<DType>)> = vec![];
    let mut atoms: Vec<Atom<DType, IType>> = vec![];

    for section in file_pair.into_inner() {
        match section.as_rule() {
            Rule::module_decl => {
                for inner in section.into_inner() {
                    match inner.as_rule() {
                        Rule::var_section => {
                            for decl in inner.into_inner().filter(|p| p.as_rule() == Rule::var_decl) {
                                let mut decl_iter = decl.into_inner();
                                let name = decl_iter.next().unwrap().as_str().to_string();
                                let dtype_rule = decl_iter.next().unwrap();
                                let dtype = match dtype_rule.as_str() {
                                    "boolean" => DType::Bool,
                                    "integer" => DType::Int,
                                    _ => DType::Bool, // placeholder
                                };
                                let wire = Wire::scalar(wires.len(), dtype.clone());
                                wires.push((name, wire));
                            }
                        }

                        Rule::assign_section => {
                            for assign in inner.into_inner().filter(|p| p.as_rule() == Rule::assign_stmt) {
                                let parts: Vec<_> = assign.into_inner().collect();
                                let target = parts[0].as_str().to_string();
                                let expr = parts[1].as_str().to_string(); // placeholder
                                println!("assign: {:?} := {:?}", target, expr);
                                // TODO: build Atom<Term> based on expr
                            }
                        }

                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    // Placeholder: build empty wires
    Module {
        external: Wire::vector(0, DType::Bool, 0),
        interface: Wire::vector(0, DType::Bool, 0),
        private: Wire::vector(0, DType::Bool, 0),
        atoms,
    }
}

fn build_assign(assign_pair: Pair<Rule>, wires: &[(String, Wire<DType>)]) -> Option<Atom<DType, IType>> {
    let mut parts = assign_pair.into_inner();
    let target = parts.next().unwrap().as_str().to_string();
    let expr_pair = parts.next().unwrap();

    let term = build_expr(expr_pair);

    // Find wire for target variable
    let write_wire = wires
        .iter()
        .find(|(n, _)| target.contains(n))
        .map(|(_, w)| w.clone())
        .unwrap_or_else(|| Wire::scalar(0, DType::Bool));

    let read_wire = write_wire.clone(); // Simplified assumption

    let atom = Atom::new_unchecked(
        read_wire.clone(),
        write_wire.clone(),
        Wire::vector(0, DType::Bool, 0),
        vec![term.clone()],
        vec![term],
    );

    Some(atom)
}

fn build_expr(expr_pair: Pair<Rule>) -> Term<DType, IType> {
    match expr_pair.as_rule() {
        Rule::expr_not => {
            let inner = expr_pair.into_inner().next().unwrap();
            let subterm = build_expr(inner);
            Term::new(
                IType::Not(Box::new(subterm.ins)), // use Box::new() and extract inner IType
                Wire::vector(0, DType::Bool, 0),
                Wire::vector(0, DType::Bool, 0),
            )
        }

        Rule::expr_and => {
            let mut inner = expr_pair.into_inner();
            let left = build_expr(inner.next().unwrap());
            let right = build_expr(inner.next().unwrap());
            Term::new(
                IType::And(Box::new(left.ins), Box::new(right.ins)), //
                Wire::vector(0, DType::Bool, 0),
                Wire::vector(0, DType::Bool, 0),
            )
        }

        Rule::expr_or => {
            let mut inner = expr_pair.into_inner();
            let left = build_expr(inner.next().unwrap());
            let right = build_expr(inner.next().unwrap());
            Term::new(
                IType::Or(Box::new(left.ins), Box::new(right.ins)), //
                Wire::vector(0, DType::Bool, 0),
                Wire::vector(0, DType::Bool, 0),
            )
        }

        Rule::expr_identifier => {
            let name = expr_pair.as_str().to_string();
            Term::new(IType::VarRef(name), Wire::vector(0, DType::Bool, 0), Wire::vector(0, DType::Bool, 0))
        }

        Rule::expr_bool => {
            let val = expr_pair.as_str().to_lowercase() == "true";
            Term::new(IType::ConstBool(val), Wire::vector(0, DType::Bool, 0), Wire::vector(0, DType::Bool, 0))
        }

        _ => panic!("unhandled expr: {:?}", expr_pair.as_rule()),
    }
}




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_model() {
        let input = r#"
            MODULE main
            VAR
              x : boolean;
              y : boolean;
            ASSIGN
              next(x) := y & !x;
        "#;

        let module = parse_nusmv(input);
        println!("Module built successfully!");
        assert!(!module.atoms.is_empty());
    }
}

