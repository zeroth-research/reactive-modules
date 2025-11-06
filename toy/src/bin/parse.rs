use std::process;

use base::module::Module;
use toy::dtype::Type;
use toy::instruction::Instruction;
type ToyModule = Module<Type, Instruction>;

use toy::parser::Parser;

use clap::Parser as ClapParser;

#[derive(ClapParser)]
struct Cli {
    // spec file for the module
    spec: String,

    // dump the parsed module to stdout
    #[arg(long, default_value_t = false)]
    stdout: bool,
}

fn main() {
    let args = Cli::parse();
    let input = std::fs::read_to_string(&args.spec).expect("Cannot read the input file");
    let mut parser: Parser = Parser::new();

    let modules = parser.parse(input);

    if modules.len() > 0 {
        println!("Modules parsed successfully!");
    }

    if args.stdout {
        for module in &modules {
            println!("{}", &module);
        }
    }
}
