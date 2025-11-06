use base::atom::Atom;
use base::term::Term;
use base::wire;
use base::wire::Wire;

pub enum LinGCL {
    Eq,
    Neq,
    Leq,
    Le,
    Geq,
    Ge,
    Sum(usize),
    Mul(usize),
    And(usize),
    Or(usize),
    Neg,
    Minus,
    MatMul,
    Choose,
    Guard,
    Tensor(Vec<f64>),
    Id,
}

#[macro_export]
macro_rules! lin_gcl {
    // // Case 1: No data, no reads
    ($variant:ident, $write:expr) => {
        base::term::Term::new(LinGCL::$variant, wire![$write], Wire::none())
    };

    // Case 2: No data, with reads
    ($variant:ident, $write:expr, $($read:expr),+ $(,)?) => {
        base::term::Term::new(LinGCL::$variant, wire!($write), wire!($($read),*))
    };

    // // Case 3: With data, no reads
    ($variant:ident($($args:expr),* $(,)?), $write:expr) => {
        base::term::Term::new(LinGCL::$variant($($args),*), wire![$write], Wire::none())
    };

    // Case 4: With data, with reads
    ($variant:ident($($args:expr),* $(,)?), $write:expr, $($read:expr),+ $(,)?) => {
        base::term::Term::new(LinGCL::$variant($($args),*), wire![$write], wire![$($read),*])
    };
}

#[test]
fn can_instantiate_example_module() {
    // w0: x
    // w1: y
    // w2: y
    // temp10: (x+y) <= z
    // temp11: x + y
    let x = wire!((0, "real"));
    let y = Wire::one(1, "real");
    //let z = Wire::one(2, "real");

    let x1 = Wire::one(5, "real");
    //let y1 = Wire::one(6, "real");
    let z1 = Wire::one(7, "real");

    let temp10 = Wire::one(10, "real");
    let xplusy = lin_gcl!(Sum(0), wire!(&temp10), wire!(&x));

    let temp11: Wire<&str> = Wire::one(11, "real");
    // // temp1 = temp0 <= z'
    let xplusyleqzp: Term<&str, LinGCL> = lin_gcl!(Leq, wire!(&temp11), wire!(&temp10, &z1));
    // // temp2 = temp0 >= z'
    let temp12 = Wire::one(12, "real");
    let xplusygeqzp = lin_gcl!(Geq, wire!(&temp12), wire!(&temp10, &z1));
    // // temp13 = 0
    let temp13 = Wire::one(13, "real");
    let temp14 = Wire::one(14, "real");
    let temp15 = Wire::one(15, "real");
    let xp1 = lin_gcl!(Tensor(vec![0.]), wire!(&temp13));
    let gc1 = lin_gcl!(Guard, wire!(&temp14), wire!(&temp11, &temp13));
    let gc2 = lin_gcl!(Guard, wire!(&temp15), wire!(&temp12, &y));
    let choose = lin_gcl!(Choose, wire!(&x1), wire!(&temp14, &temp15));

    let _atom: Atom<&str, LinGCL> = Atom::new_unchecked(
        wire![x, y],
        wire![x1],
        wire![z1],
        vec![],
        vec![xplusy, xplusyleqzp, xplusygeqzp, xp1, gc1, gc2, choose],
    );
}
