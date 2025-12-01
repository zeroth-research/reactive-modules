mod context;
mod term;

#[cfg(feature = "visual-html")]
mod html;

pub use context::Context;

pub use crate::term::TorchDType as DType;
pub use crate::term::TorchOp as IType;
pub use crate::term::TorchTerm;
pub type TorchModule = base::module::Module<DType, IType>;
pub type TorchAtom = base::atom::Atom<DType, IType>;
