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

    // how to dump the module, e.g., --dump html
    #[arg(long)]
    dump: Option<String>,

    // open dump module (if module is dumped)
    #[arg(long, default_value_t = false)]
    open: bool,

    // dump the parsed module to stdout
    #[arg(long, default_value_t = false)]
    stdout: bool,
}

#[cfg(feature = "visual-html")]
fn dump_to_html(modules: &Vec<ToyModule>, open: bool) {
    todo!();
}

#[cfg(not(feature = "visual-html"))]
fn dump_to_html(modules: &Vec<ToyModule>, open: bool) {
    eprintln!("HTML visualization is disabled, enable the feature \"visual-html\"".);
    process::exit(1);
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
            println!("{:?}", &module);
        }
    }

    if let Some(method) = &args.dump {
        match method.as_str() {
            "html" | "HTML" => {
                dump_to_html(&modules, args.open);
            }
            _ => {
                eprint!("Unknown `dump` method.");
                process::exit(1);
            }
        }
    }
}
