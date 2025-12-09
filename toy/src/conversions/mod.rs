mod smt;

// we must wrap ToyModule, because we cannot define traits for it otherwise (it is foreign from the
// `base` package)
pub struct ModuleConverter<'a>(pub &'a crate::ToyModule);
