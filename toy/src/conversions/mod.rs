#[cfg(feature = "conversions-smt")]
mod smt;

// we must wrap ToyModule, because we cannot define traits for it otherwise (it is foreign from the
// `base` package)
pub struct ModuleConverter<'a>(pub &'a crate::ToyModule);

#[cfg(feature = "conversions-smt")]
pub fn to_smt(
    module: &crate::ToyModule,
) -> Result<base::Module<::smt::dtype::DType, ::smt::itype::IType>, &'static str> {
    TryInto::<base::Module<::smt::dtype::DType, ::smt::itype::IType>>::try_into(ModuleConverter(
        module,
    ))
}
