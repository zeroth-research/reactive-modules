use base::{Interface, Module, Term, Wire, term};
use common::context::Context;
use common::transition::WiredTransitions;
use common::unrolling::ModuleUnrolling;

#[allow(clippy::vec_init_then_push)]
fn example_counter() -> Result<Module<&'static str, &'static str>, &'static str> {
    let x0 = Wire::new(0, "real");
    let y0 = Wire::new(1, "real");
    let z0 = Wire::new(2, "real");
    let y00 = Wire::new(3, "real");
    let z00 = Wire::new(4, "real");
    let x1 = Wire::new(5, "real");
    let y1 = Wire::new(6, "real");
    let z1 = Wire::new(7, "real");
    let y01 = Wire::new(8, "real");
    let z01 = Wire::new(9, "real");

    let mut init: Vec<Term<&str, &str>> = Vec::new();

    init.push(term!("ZERO", [(10, "real")])?);

    init.push(term!("ID", [x1.clone()], [(10, "real")])?);
    init.push(term!("ABS", [(11, "bool")], [y01.clone()])?);
    init.push(term!("ID", [y1.clone()], [(11, "bool")])?);
    init.push(term!("ABS", [(12, "bool")], [z01.clone()])?);
    init.push(term!("ID", [z1.clone()], [(12, "bool")])?);

    let mut update: Vec<Term<&str, &str>> = Vec::new();

    update.push(term!("ZERO", [(10, "real")])?);
    update.push(term!("LEQ", [(13, "bool")], [x0.clone(), y0.clone()])?);
    update.push(term!("LEQ", [(14, "bool")], [x0.clone(), z0.clone()])?);
    update.push(term!("OR", [(15, "bool")], [(13, "bool"), (14, "bool")])?);

    update.push(term!("ONE", [(16, "real")])?);
    update.push(term!(
        "ADD",
        [(17, "real")],
        [x0.clone(), (16, "real").into()]
    )?);

    update.push(term!(
        "ITE",
        [x1.clone()],
        [(15, "bool"), (17, "real"), (10, "real")]
    )?);
    update.push(term!("ID", [y1.clone()], [y0.clone()])?);
    update.push(term!("ID", [z1.clone()], [z0.clone()])?);

    let obs = Interface::from_iter([[x0, x1], [y0, y1], [z0, z1], [y00, y01], [z00, z01]]);

    Module::sequential_observable(obs, init, update)
}

#[test]
fn unroll_1() {
    let module = example_counter().unwrap();

    let mut ctx = Context::from_module(&module);
    let mut unrolling: WiredTransitions<&str, &str> = WiredTransitions::new();
    let mut unroll = ModuleUnrolling::new(&module, &mut ctx);

    unrolling = unroll.init(unrolling);

    for _ in 0..5 {
        unrolling = unroll.step(unrolling);
        //dbg!(&state);
    }

    for transition in unrolling.transitions {
        println!("---------------------------");
        println!("In: {}", transition.intf_in());
        println!("---------------------------");
        for term in &transition {
            println!("{}", term);
        }
        println!("---------------------------");
        println!("Out: {}", transition.intf_out());
    }
}
