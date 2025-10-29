use base::wire::Wire;

use toy::instruction::Instruction;
use toy::term::Term;
use toy::val::{Type, Val};

#[cfg(test)]
mod tests {
    use super::*;

    fn init(vars: &[Wire<Type>; 10]) -> Vec<Term> {
        let init_x = Term::new(
            Instruction::Const(Val::Int(0)),
            Wire::none(),
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
        let wire10 = Wire::one(10, Type::Bool);
        let wire11 = Wire::one(11, Type::Bool);
        let wire12 = Wire::one(11, Type::Bool);
        let wire13 = Wire::one(11, Type::Bool);
        let xlty = Term::new(Instruction::Lt, wire10.clone(), reads);

        vec![xlty]
    }

    #[test]
    fn toy_example() {
        // build module
        let vars = [
            Wire::scalar(0, Type::Int),
            Wire::scalar(1, Type::Int),
            Wire::scalar(2, Type::Int),
            Wire::scalar(3, Type::NInt),
            Wire::scalar(4, Type::NInt),
            Wire::scalar(5, Type::Int),
            Wire::scalar(6, Type::Int),
            Wire::scalar(7, Type::Int),
            Wire::scalar(8, Type::NInt),
            Wire::scalar(9, Type::NInt),
        ];

        let init_terms = init(&vars);
        let update_terms = update(&vars);
    }
}
