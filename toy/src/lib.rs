pub mod context;
pub mod dtype;
pub mod interpreter;
pub mod itype;
pub mod mat;
pub mod parser;
pub mod term;
pub mod val;

#[cfg(feature = "visual")]
pub mod visual;

pub type DType = crate::dtype::DType;
pub type IType = crate::itype::IType;
pub type ToyModule = base::module::Module<DType, IType>;
pub type ToyAtom = base::atom::Atom<DType, IType>;
pub type ToyTerm = base::term::Term<DType, IType>;
pub type ToyWire = base::wire::Wire<DType>;
