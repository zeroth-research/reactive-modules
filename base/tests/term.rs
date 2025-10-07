use base;
use base::term::Instruction;

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

impl Instruction for LinGCL {
    fn arity(&self) -> Option<usize> {
        Some(match self {
            LinGCL::Eq | LinGCL::Neq | LinGCL::Leq | LinGCL::Le | LinGCL::Geq | LinGCL::Ge => 2,
            LinGCL::Sum(_) | LinGCL::Mul(_) | LinGCL::And(_) | LinGCL::Or(_) => 2,
            LinGCL::Neg | LinGCL::Minus => 1,
            LinGCL::MatMul => 2,
            LinGCL::Choose => None?,
            LinGCL::Guard => 2,
            LinGCL::Tensor(_) => 0,
            LinGCL::Id => 1,
        })
    }
}

#[macro_export]
macro_rules! lin_gcl {
    // Case 1: No data, no reads
    ($variant:ident, $write:expr) => {
        base::term::Term::new(LinGCL::$variant, wires![$write], wires![])
    };

    // Case 2: No data, with reads
    ($variant:ident, $write:expr, $($read:expr),+ $(,)?) => {
        base::term::Term::new(LinGCL::$variant, wires![$write], wires![$($read),*])
    };

    // Case 3: With data, no reads
    ($variant:ident($($args:expr),* $(,)?), $write:expr) => {
        base::term::Term::new(LinGCL::$variant($($args),*), wires![$write], wires![])
    };

    // Case 4: With data, with reads
    ($variant:ident($($args:expr),* $(,)?), $write:expr, $($read:expr),+ $(,)?) => {
        base::term::Term::new(LinGCL::$variant($($args),*), wires![$write], wires![$($read),*])
    };
}

#[test]
fn can_instantiate_example_module() {
    // temp1 = (x+y) <= z'
    // temp0 = x + y
    // let xplusy = lin_gcl!(Sum(0), "temp0", "x", "y");
    // // temp1 = temp0 <= z'
    // let xplusyleqzp = lin_gcl!(Leq, "temp1", "temp0", "z'");
    // // temp2 = temp0 >= z'
    // let xplusygeqzp = lin_gcl!(Geq, "temp2", "temp0", "z'");
    // // temp3 = 0
    // let xp1 = lin_gcl!(Tensor(vec![0.]), "temp3");
    // let gc1 = lin_gcl!(Guard, "temp4", "temp1", "temp3");
    // let gc2 = lin_gcl!(Guard, "temp5", "temp2", "y");
    // let choose = lin_gcl!(Choose, "x'", "temp4", "temp5");
    //
    // let atom = Atom::new_unchecked(
    //     wires!["x", "y"],
    //     wires!["x'"],
    //     wires!["z'"],
    //     vec![],
    //     vec![xplusy, xplusyleqzp, xplusygeqzp, xp1, gc1, gc2, choose],
    // );
}
