use base::module::Module;
//use std::fs::metadata;
use std::process;
use toy::context::Context;
use toy::dtype::Type;
use toy::instruction::Instruction;
type ToyModule = Module<Type, Instruction>;

use toy::parser::Parser;

use clap::Parser as ClapParser;

#[cfg(feature = "visual-html")]
use visual::html;

#[derive(ClapParser)]
struct Cli {
    // spec file for the module
    spec: String,

    // how to dump the module, e.g., --dump html
    #[arg(long)]
    dump: Option<String>,

    // output file/dir
    // #[arg(long)]
    // output: Option<String>,

    // open dump module (if module is dumped)
    #[arg(long, default_value_t = false)]
    open: bool,

    // dump the parsed module to stdout
    #[arg(long, default_value_t = false)]
    stdout: bool,
}

#[cfg(feature = "visual-html")]
fn dump_to_html(modules: &[ToyModule], args: &Cli, ctx: &Context) -> Result<(), std::io::Error> {
    // TODO: enable output to cusom file/dir
    for (n, module) in modules.iter().enumerate() {
        let module_name: Option<String> = None; //module.name();
        let path = if let Some(name) = module_name {
            format!("{}.{}.html", args.spec, name)
        } else {
            format!("{}.module-{}.html", args.spec, n)
        };

        html::write_to_html(module, path.as_str(), Some(ctx))
            .inspect_err(|err| {
                eprintln!("Failed writing the module to file {}", path);
                eprintln!("{}", err)
            })
            .expect("Failed generating HTML");

        println!("Module written to `{}`", path);

        if args.open {
            println!("Openning in web browser...");
            #[cfg(target_os = "macos")]
            {
                process::Command::new("open")
                    .arg(path)
                    .spawn()
                    .unwrap()
                    .wait()?;
            }
            #[cfg(target_os = "linux")]
            {
                process::Command::new("xdg-open")
                    .arg(path)
                    .spawn()
                    .unwrap()
                    .wait()?;
            }
        }
    }

    Ok(())
}

#[cfg(not(feature = "visual-html"))]
fn dump_to_html(modules: &Vec<ToyModule>, args: &Cli) {
    eprintln!("HTML visualization is disabled, enable the feature \"visual-html\"");
    process::exit(1);
}

fn main() {
    let args = Cli::parse();
    let input = std::fs::read_to_string(&args.spec).expect("Cannot read the input file");
    let mut parser: Parser = Parser::new();

    let modules = parser.parse(input);

    if !modules.is_empty() {
        println!("Modules parsed successfully!");
    }

    if args.stdout {
        for module in &modules {
            println!("{}", &module);
        }
    }

    if let Some(method) = &args.dump {
        match method.as_str() {
            "html" | "HTML" => {
                if let Err(e) = dump_to_html(&modules, &args, parser.ctx()) {
                    eprint!("Failed dumping to HTML: {}", e);
                }
            }
            _ => {
                eprint!("Unknown `dump` method.");
                process::exit(1);
            }
        }
    }
}
