use toy::ToyContext;
use toy::dtype::DType;
use toy::interpreter::Interpreter;

mod example_module;

use example_module::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toy_example() {
        let mut ctx = ToyContext::new();

        // create variables
        ctx.intf(
            DType::Int,
            &["x", "y", "z", "y0", "z0", "x'", "y'", "z'", "y0'", "z0'"],
        );

        // build module
        let module = build_module(&mut ctx);
        let _interpreter = Interpreter::new(&module);
    }
}
