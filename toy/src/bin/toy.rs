//use std::fs::metadata;
use std::process;

use toy::parser::Parser;

use clap::Parser as ClapParser;

#[cfg(feature = "visual-html")]
use visual::html;
#[cfg(feature = "visual-html")]
use visual::html::Descriptor;

#[cfg(feature = "conversions-smt")]
use toy::conversions::ModuleConverter;

#[derive(ClapParser)]
struct Cli {
    // spec file for the module
    spec: String,

    // how to dump the module, e.g., --dump html
    #[arg(long)]
    dump: Option<String>,

    // output file/dir
    #[arg(long)]
    to: Option<String>,

    // open dump module (if module is dumped)
    #[arg(long, default_value_t = false)]
    open: bool,

    // dump the parsed module to stdout
    #[arg(long, default_value_t = false)]
    stdout: bool,
}

#[cfg(feature = "visual-html")]
fn dump_to_html<D, I, C>(
    module: &base::Module<D, I>,
    module_name: &str,
    args: &Cli,
    ctx: Option<&C>,
) -> Result<(), std::io::Error>
where
    I: std::fmt::Display,
    D: std::fmt::Display,
    C: Descriptor<D, I>,
{
    let path = format!("{}.{}.html", args.spec, module_name);

    html::write_to_html(module, path.as_str(), ctx)
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

    Ok(())
}

fn dump_module<D, I, C>(module: &base::Module<D, I>, idx: usize, args: &Cli, descr: Option<C>)
where
    I: std::fmt::Display,
    D: std::fmt::Display + std::cmp::Eq + Copy,
    C: Descriptor<D, I>,
{
    // get or create the name of the module for displaying
    let module_name: Option<String> = None; // module.name();
    let module_name = if let Some(name) = module_name {
        name.to_string()
    } else {
        format!("Module {idx}")
    };

    // dump the module to stdout if required
    if args.stdout {
        println!("## Module `{}`", &module_name);
        println!("{}", &module);
    }

    #[cfg(feature = "visual-html")]
    if let Some(method) = &args.dump {
        match method.as_str() {
            "html" | "HTML" => {
                if let Err(e) = dump_to_html(module, module_name.as_str(), args, descr.as_ref()) {
                    eprint!("Failed dumping to HTML: {}", e);
                }
            }
            _ => {
                eprint!("Unknown `dump` method: {method}.");
                process::exit(-1);
            }
        }
    }
}

fn main() {
    let args = Cli::parse();
    let input = std::fs::read_to_string(&args.spec).expect("Cannot read the input file");
    let mut parser: Parser = Parser::new();

    let modules = parser.parse(input);

    if !modules.is_empty() {
        println!("Modules parsed successfully!");
    }

    for (n, module) in modules.iter().enumerate() {
        if let Some(to) = &args.to {
            match to.as_str() {
                #[cfg(feature = "conversions-smt")]
                "smt" => {
                    let module = ModuleConverter(module).try_into();
                    let module = match module {
                        Err(e) => {
                            eprintln!("Failed translating module to {to}: {e}");
                            process::exit(-1)
                        }
                        Ok(m) => m,
                    };
                    let ctx: Option<smt::html::Context> = None;
                    dump_module(&module, n, &args, ctx);
                }
                _ => {
                    eprintln!("Invalid translation type. Do not know how to translate to `{to}`");
                    process::exit(-1)
                }
            }
        } else {
            dump_module(
                module,
                n,
                &args,
                Some(toy::visual::html::HTMLDescriptor::new(parser.ctx())),
            );
        }
    }
}
