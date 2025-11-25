use smt::dtype::DType;
use smt::itype::IType;
use smt::html::SmtDescriptor;

use base::module::Module;
use visual::html;

use clap::Parser as ClapParser;
use std::process;

#[derive(ClapParser)]
struct Cli {
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
        let module_name = module.name();
        let path = if let Some(name) = module_name {
            format!("{}.html", name)
        } else {
            format!("smt/tests/module-{}.html", n)
        };

        let descr = SmtDescriptor::new(module);
        html::write_to_html(module, path.as_str(), Some(&descr))
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

// cargo run --bin parse --package smt -- --dump html --open
fn main() {
    let args = Cli::parse();

    let module = smt::create_smt_module::create_test_module();
    let modules = vec![module];

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
                // dump_to_html(&modules, &args, parser.ctx());
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

    //TODO: call the SMT parser (which will be in smt/src/smt.rs) and generate the SMT file
    smt::smt::parse_modules(&modules);
}