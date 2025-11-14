use base::module::Module;
use base::term::TermWire;
use base::{Term, Wire};
use smv::dtype::DType;
use smv::itype::IType;
use std::{fs, process};

use clap::Parser as ClapParser;
use smv::lean::{collect_used_vars, render_terms_to_lean, to_lean_from_module};
use smv::smv::parse_smv;

use smv::html::SmvDescriptor;
use visual::html;

fn build_obligations() -> (
    Vec<Term<DType, IType>>, // invariant
    Vec<Term<DType, IType>>, // variant
    Vec<Term<DType, IType>>, // buchi
) {
    // invariant: x0 <= x1 ∨ x0 <= x2
    let invariant: Vec<Term<DType, IType>> = vec![
        Term::new(
            IType::Le,
            Wire::one(3, DType::Bool),
            Wire::many(0, DType::Int, 2),
        ),
        Term::new(
            IType::Le,
            Wire::one(4, DType::Bool),
            Wire::one(0, DType::Int).extend(&Wire::one(2, DType::Int)),
        ),
        Term::new(
            IType::Or,
            Wire::one(5, DType::Bool),
            Wire::many(3, DType::Bool, 2),
        ),
    ];

    // variant: reluZ (x1 - x0) + reluZ (x2 - x0)
    // We construct two Sub terms that compute x1 - x0 and x2 - x0 (note the
    // operand order), then conditionals implementing relu (if diff < 0 then 0
    // else diff), and finally an Add of the two relu results.
    let variant: Vec<Term<DType, IType>> = vec![
        // diff1 := x1 - x0
        Term::new(
            IType::Sub,
            Wire::one(3, DType::Int),
            // order: x1, x0
            Wire::one(1, DType::Int).extend(&Wire::one(0, DType::Int)),
        ),
        // diff2 := x2 - x0
        Term::new(
            IType::Sub,
            Wire::one(4, DType::Int),
            // order: x2, x0
            Wire::one(2, DType::Int).extend(&Wire::one(0, DType::Int)),
        ),
        // const 0 at idx 5
        Term::new(IType::ConstInt(0), Wire::one(5, DType::Int), Wire::none()),
        // diff1 < 0 -> idx 6
        Term::new(
            IType::Lt,
            Wire::one(6, DType::Bool),
            Wire::one(3, DType::Int).extend(&Wire::one(5, DType::Int)),
        ),
        // diff2 < 0 -> idx 7
        Term::new(
            IType::Lt,
            Wire::one(7, DType::Bool),
            Wire::one(4, DType::Int).extend(&Wire::one(5, DType::Int)),
        ),
        // cond1: if diff1 < 0 then 0 else diff1 -> idx 8
        Term::new(
            IType::Cond,
            Wire::one(8, DType::Int),
            Wire::one(6, DType::Bool)
                .extend(&Wire::one(5, DType::Int))
                .extend(&Wire::one(3, DType::Int)),
        ),
        // cond2: if diff2 < 0 then 0 else diff2 -> idx 9
        Term::new(
            IType::Cond,
            Wire::one(9, DType::Int),
            Wire::one(7, DType::Bool)
                .extend(&Wire::one(5, DType::Int))
                .extend(&Wire::one(4, DType::Int)),
        ),
        // add relu1 + relu2 -> idx 10
        Term::new(
            IType::Add,
            Wire::one(10, DType::Int),
            Wire::many(8, DType::Int, 2),
        ),
    ];

    // buchi condition: x0 = x1 ∨ x0 = x2
    let buchi: Vec<Term<DType, IType>> = vec![
        Term::new(
            IType::Eq,
            Wire::one(3, DType::Bool),
            Wire::many(0, DType::Int, 2),
        ),
        Term::new(
            IType::Eq,
            Wire::one(4, DType::Bool),
            Wire::one(0, DType::Int).extend(&Wire::one(2, DType::Int)),
        ),
        Term::new(
            IType::Or,
            Wire::one(5, DType::Bool),
            Wire::many(3, DType::Bool, 2),
        ),
    ];

    (invariant, variant, buchi)
}

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
    let lean = to_lean_from_module(&module).unwrap();
    let (invariant, variant, buchi) = build_obligations();
    let wire_pair = module.wire();
    let latched = &wire_pair[0];
    let mut wires_map: Vec<(String, usize, DType)> = Vec::new();
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (i, dtype) in latched.iter() {
        if seen.insert(i) {
            wires_map.push((format!("x{}", i), i, *dtype));
        }
    }
    let var_count = wires_map.len().saturating_sub(module.extl()[0].len());

    let inv_s = render_terms_to_lean(&invariant, &invariant.write(), &wires_map, var_count);
    let var_s = render_terms_to_lean(&variant, &variant.write(), &wires_map, var_count);
    let buch_s = render_terms_to_lean(&buchi, &buchi.write(), &wires_map, var_count);

    // Compute variable tuples for each obligation by listing used VAR indices.
    let buch_vars = collect_used_vars(&buchi, &buchi.write(), var_count);
    let inv_vars = collect_used_vars(&invariant, &invariant.write(), var_count);
    let var_vars = collect_used_vars(&variant, &variant.write(), var_count);
    let buch_tuple = buch_vars
        .iter()
        .map(|i| format!("x{}", i))
        .collect::<Vec<_>>()
        .join(", ");
    let inv_tuple = inv_vars
        .iter()
        .map(|i| format!("x{}", i))
        .collect::<Vec<_>>()
        .join(", ");
    let var_tuple = var_vars
        .iter()
        .map(|i| format!("x{}", i))
        .collect::<Vec<_>>()
        .join(", ");

    let obligations = format!(
        "def buchi_condition : State → Prop :=\n  fun ⟨{}⟩ ↦\n    {}\n\ndef invariant : State → Prop :=\n  fun ⟨{}⟩ ↦\n    {}\n\ndef relu : Int → Int := fun x ↦ max x 0\n\ndef variant : State → Int :=\n  fun ⟨{}⟩ ↦\n    {}",
        buch_tuple, buch_s, inv_tuple, inv_s, var_tuple, var_s
    );

    let full = format!("{}\n{}", lean, obligations);

    let out_path = format!("{}/tests/counter.lean", env!("CARGO_MANIFEST_DIR"));
    std::fs::create_dir_all(format!("{}/tests", env!("CARGO_MANIFEST_DIR"))).unwrap();
    fs::write(out_path, full).unwrap();

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
