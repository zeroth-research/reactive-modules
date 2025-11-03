use toy::context::Context;
use toy::dtype::Type;
use toy::interpreter::{Interpreter, eval};

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
            Type::Int,
            &["x", "y", "z", "y0", "z0", "x'", "y'", "z'", "y0'", "z0'"],
        );

        // build module
        let module = build_module(&mut ctx);
        let interpreter = Interpreter::new(&module);
    }
}
