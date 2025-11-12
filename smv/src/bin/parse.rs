use std::process;
use base::module::Module;
use smv::dtype::DType;
use smv::itype::IType;

use clap::Parser as ClapParser;
use smv::smv::parse_smv;

use visual::html;
use smv::html::SmvDescriptor;

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

fn dump_to_html(modules: &Vec<Module<DType, IType>>, args: &Cli) {
    for (n, module) in modules.iter().enumerate() {
        let module_name = module.name();
        let path = if module_name.is_some() {
            format!("{}.{}.html", args.spec, module_name.unwrap())
        } else {
            format!("{}.module-{}.html", args.spec, n)
        };

        // Use our SMV descriptor for nicer HTML descriptions
        let descr = SmvDescriptor::new();
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
                process::Command::new("open").arg(path).spawn().unwrap();
            }
            #[cfg(target_os = "linux")]
            {
                process::Command::new("xdg-open").arg(path).spawn().unwrap();
            }
        }
    }
}

// cargo run --bin parse --package smv smv/tests/counter.smv --dump html --open
fn main() {
    let args = Cli::parse();
    let input = std::fs::read_to_string(&args.spec).expect("Cannot read the input file");

    let module = parse_smv(&input).unwrap();
    let modules = vec![module];

    if modules.len() > 0 {
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
                dump_to_html(&modules, &args);
            }
            _ => {
                eprint!("Unknown `dump` method.");
                process::exit(1);
            }
        }
    }
}