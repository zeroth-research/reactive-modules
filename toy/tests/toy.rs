use toy::dtype::DType;

use toy::context::Context;

mod example_module;

use example_module::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toy_example() {
        let mut ctx = Context::new();

        // create variables
        ctx.vars(
            DType::Int,
            &["x", "y", "z", "y0", "z0", "x'", "y'", "z'", "y0'", "z0'"],
        );

        // build module
        let module = build_module(&mut ctx);
        dbg!(module);

        let _prop = _build_prop(&mut ctx);
        //dbg!(prop);
    }

    #[cfg(feature = "conversions-smt")]
    #[test]
    fn conversions_smt1() {
        let mut ctx = Context::new();
        ctx.vars(
            DType::Int,
            &["x", "y", "z", "y0", "z0", "x'", "y'", "z'", "y0'", "z0'"],
        );

        // build module
        let module = build_module(&mut ctx);
        let _smt_module: base::Module<smt::dtype::DType, smt::itype::IType> =
            toy::conversions::ModuleConverter(&module)
                .try_into()
                .unwrap();
    }
}
