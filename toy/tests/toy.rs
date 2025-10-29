use base::wire::Wire;

use toy::instruction::Instruction;
use toy::term::{Term, construct};
use toy::val::{Type, Val};

#[cfg(test)]
mod tests {
    use super::*;
    use base::atom::Atom;
    use base::module::Module;

    fn init(vars: &[Wire<Type>; 10]) -> Vec<Term> {
        let init_x = construct(
            Instruction::Const(Val::Int(0)),
            vars[5].clone(),
            Wire::none(),
        )
        .expect("Failed creating term");
        let init_y = construct(Instruction::Id, vars[6].clone(), vars[8].clone())
            .expect("Failed creating term");
        let init_z = construct(Instruction::Id, vars[7].clone(), vars[9].clone())
            .expect("Failed creating term");

        vec![init_x, init_y, init_z]
    }

    fn update(vars: &[Wire<Type>; 10]) -> Vec<Term> {
        // wire10 = x < y
        let reads = vars[0].union(&vars[1]).unwrap();
        let wire10 = Wire::one(10, Type::Bool);
        let xlty = construct(Instruction::Lt, wire10.clone(), reads).expect("Failed creating term");

        // wire11 = x < z
        let reads = vars[0].union(&vars[2]).unwrap();
        let wire11 = Wire::one(11, Type::Bool);
        let xltz = construct(Instruction::Lt, wire11.clone(), reads).expect("Failed creating term");

        // wire12 = wire10 || wire11
        let wire12 = Wire::one(12, Type::Bool);
        let reads = wire10.union(&wire11).unwrap();
        let or = construct(Instruction::Or, wire12.clone(), reads).expect("Failed creating term");

        // zero
        let const0 = Wire::one(13, Type::Int);
        let term0 = construct(
            Instruction::Const(Val::Int(0)),
            const0.clone(),
            Wire::none(),
        )
        .expect("Failed creating term");

        // one
        let const1 = Wire::one(14, Type::Int);
        let term1 = construct(
            Instruction::Const(Val::Int(1)),
            const1.clone(),
            Wire::none(),
        )
        .expect("Failed creating term");

        // wire15 = vars[0] + const1
        let wire15 = Wire::one(15, Type::Int);
        let reads = vars[0].union(&const1).unwrap();
        let sum = construct(Instruction::Sum, wire15.clone(), reads).expect("Failed creating term");

        // wire5 = ite(wire12, wire15, const0)
        let reads = wire12.union(&wire15).unwrap().union(&const0).unwrap();
        let ite =
            construct(Instruction::Ite, vars[5].clone(), reads).expect("Failed creating term");

        // y' := y
        let id_y = construct(Instruction::Id, vars[6].clone(), vars[1].clone())
            .expect("Failed creating term");
        let id_z = construct(Instruction::Id, vars[7].clone(), vars[2].clone())
            .expect("Failed creating term");

        vec![xlty, xltz, or, term0, term1, sum, ite, id_y, id_z]
    }

    #[test]
    fn toy_example() {
        // build module
        let vars = [
            // x y z y0 z0
            Wire::one(0, Type::Int), // x
            Wire::one(1, Type::Int), // y
            Wire::one(2, Type::Int), // z
            Wire::one(3, Type::Int), // y0
            Wire::one(4, Type::Int), // z0
            // primed variables
            Wire::one(5, Type::Int), // x'
            Wire::one(6, Type::Int), // y'
            Wire::one(7, Type::Int), // z'
            Wire::one(8, Type::Int), // y0'
            Wire::one(9, Type::Int), // z0'
        ];

        let init_terms = init(&vars);
        let update_terms = update(&vars);

        const NEXT_OFFSET: isize = 5;
        // x y z
        let read = Wire::many(0, Type::Int, 3);
        // y0 z0
        let wait = Wire::many(3, Type::Int, 2);

        // x y z y0 z0
        let latched = read.union(&wait).expect("Failed creating union");
        // x' y' z' y0' z0'
        let next = latched.twin(NEXT_OFFSET).expect("Failed getting twins");

        let atom =
            Atom::with_module_wire(&[latched.clone(), next.clone()], init_terms, update_terms)
                .expect("failed creating atom");

        let module = Module::with_atoms([latched, next], vec![atom]);

        dbg!(module);
        //let atom = Atom::new_unchecked(ctrl, wait, read, init, update)
    }
}
