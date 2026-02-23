use base::module::Module;
use smv::dtype::DType;
use smv::itype::IType;
use std::process;

use clap::Parser as ClapParser;
use smv::smv::parse_smv;

use smv::html::SmvDescriptor;
use visual::html;

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

fn dump_to_html(modules: &[Module<DType, IType>], args: &Cli) -> Result<(), std::io::Error> {
    for (n, module) in modules.iter().enumerate() {
        let module_name: Option<&str> = None; // Module name not available in new API
        let path = if let Some(name) = module_name {
            format!("{}.{}.html", args.spec, name)
        } else {
            format!("{}.module-{}.html", args.spec, n)
        };

        // Use our SMV descriptor for nicer HTML descriptions
        let descr = SmvDescriptor::new(module);
        html::module::write_to_html(module, path.as_str(), Some(&descr))
            .inspect_err(|err| {
                eprintln!("Failed writing the module to file {}", path);
                eprintln!("{}", err)
            })
            .expect("Failed generating HTML");

        println!("Module written to `{}`", path);

        if args.open {
            println!("Opening in web browser...");
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

// cargo run --bin parse --package smv smv/tests/counter.smv --dump html --open
fn main() {
    let args = Cli::parse();
    let input = std::fs::read_to_string(&args.spec).expect("Cannot read the input file");

    let module = parse_smv(&input).unwrap();
    let modules = vec![module];

    println!("Modules parsed successfully!");

    if args.stdout {
        for module in &modules {
            println!("{}", &module);
        }
    }

    if let Some(method) = &args.dump {
        match method.as_str() {
            "html" | "HTML" => {
                if let Err(msg) = dump_to_html(&modules, &args) {
                    eprintln!("Failed dumping module to HTML: {msg}");
                }
            }
            _ => {
                eprint!("Unknown `dump` method.");
                process::exit(1);
            }
        }
    }
}
