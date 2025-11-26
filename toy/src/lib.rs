pub mod context;
pub mod dtype;
pub mod instruction;
pub mod interpreter;
pub mod mat;
pub mod parser;
pub mod term;
pub mod val;

#[cfg(feature = "visual")]
pub mod visual;

pub type ToyModule = base::module::Module<crate::dtype::Type, crate::instruction::Instruction>;
pub type ToyAtom = base::atom::Atom<crate::dtype::Type, crate::instruction::Instruction>;
pub type ToyTerm = base::term::Term<crate::dtype::Type, crate::instruction::Instruction>;
pub type ToyWire = base::wire::Wire<crate::dtype::Type>;
