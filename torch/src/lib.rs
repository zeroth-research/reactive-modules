pub mod context;
mod term;

#[cfg(feature = "visual-html")]
mod html;

pub use context::Context;

pub use crate::term::DType;
pub use crate::term::IType;
pub use crate::term::TorchTerm;
pub type TorchModule = base::module::Module<DType, IType>;
pub type TorchAtom = base::atom::Atom<DType, IType>;
