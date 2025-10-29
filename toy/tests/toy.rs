use base::atom::Atom;
use base::module::Module;
use base::wire::Wire;

use toy::instruction::Instruction;
use toy::term::Term;
use toy::val::{Type, Val};

#[cfg(test)]
mod tests {
    use super::*;

    fn init(vars: &[Wire<Type>; 10]) -> Vec<Term> {
        let init_x = Term::new(
            Instruction::Const(Val::Real(0.0)),
            Wire::empty(),
            vars[5].clone(),
        );
        let init_y = Term::new(Instruction::Id, vars[1].clone(), vars[8].clone());
        let init_z = Term::new(Instruction::Id, vars[7].clone(), vars[9].clone());

        vec![init_x, init_y, init_z]
    }

    fn update(vars: &[Wire<Type>; 10]) -> Vec<Term> {
        let reads = vars[0]
            .union(&vars[1])
            .expect("Failed creating termx x < y");
        let wire10 = Wire::scalar(10, Type::Bool);
        let xlty = Term::new(Instruction::Lt, wire10.clone(), reads);

        vec![xlty]
    }

    #[test]
    fn toy_example() {
        // build module
        let vars = [
            Wire::scalar(0, Type::Real),
            Wire::scalar(1, Type::Real),
            Wire::scalar(2, Type::Real),
            Wire::scalar(3, Type::NReal),
            Wire::scalar(4, Type::NReal),
            Wire::scalar(5, Type::Real),
            Wire::scalar(6, Type::Real),
            Wire::scalar(7, Type::Real),
            Wire::scalar(8, Type::NReal),
            Wire::scalar(9, Type::NReal),
        ];

        let init_terms = init(&vars);
        let update_terms = update(&vars);
    }
}
