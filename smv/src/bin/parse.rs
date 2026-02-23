use base::module::Module;
use base::{Term, Wire};
use smv::dtype::DType;
use smv::itype::IType;
use std::{fs, process};

use clap::Parser as ClapParser;
use smv::lean::{collect_used_vars, render_terms_to_lean, to_lean_from_module};
use smv::smv::parse_smv;

use smv::html::SmvDescriptor;
use visual::html;

type TermType = Term<DType, IType>;

fn build_obligations() -> (
    Vec<TermType>, // invariant
    Vec<TermType>, // variant
    Vec<TermType>, // buchi
) {
    // invariant: x0 <= x1 ∨ x0 <= x2
    let invariant: Vec<Term<DType, IType>> = vec![
        Term::function(
            IType::Le,
            [Wire::new(3, DType::Bool)],
            vec![Wire::new(0, DType::Int); 2],
        ).unwrap(),
        Term::function(
            IType::Le,
            [Wire::new(4, DType::Bool)],
            vec![Wire::new(0, DType::Int), Wire::new(2, DType::Int)],
        ).unwrap(),
        Term::function(
            IType::Or,
            [Wire::new(5, DType::Bool)],
            vec![Wire::new(3, DType::Bool); 2],
        ).unwrap(),
    ];

    // variant: reluZ (x1 - x0) + reluZ (x2 - x0)
    // We construct two Sub terms that compute x1 - x0 and x2 - x0 (note the
    // operand order), then conditionals implementing relu (if diff < 0 then 0
    // else diff), and finally an Add of the two relu results.
    let variant: Vec<Term<DType, IType>> = vec![
        // diff1 := x1 - x0
        Term::function(
            IType::Sub,
            [Wire::new(3, DType::Int)],
            // order: x1, x0
            vec![Wire::new(1, DType::Int), Wire::new(0, DType::Int)],
        ).unwrap(),
        // diff2 := x2 - x0
        Term::function(
            IType::Sub,
            [Wire::new(4, DType::Int)],
            // order: x2, x0
            vec![Wire::new(2, DType::Int), Wire::new(0, DType::Int)],
        ).unwrap(),
        // const 0 at idx 5
        Term::function::<Wire<DType>, Wire<DType>, _, _>(IType::ConstInt(0), [Wire::new(5, DType::Int)], vec![]).unwrap(),
        // diff1 < 0 -> idx 6
        Term::function(
            IType::Lt,
            [Wire::new(6, DType::Bool)],
            vec![Wire::new(3, DType::Int), Wire::new(5, DType::Int)],
        ).unwrap(),
        // diff2 < 0 -> idx 7
        Term::function(
            IType::Lt,
            [Wire::new(7, DType::Bool)],
            vec![Wire::new(4, DType::Int), Wire::new(5, DType::Int)],
        ).unwrap(),
        // cond1: if diff1 < 0 then 0 else diff1 -> idx 8
        Term::function(
            IType::Cond,
            [Wire::new(8, DType::Int)],
            vec![Wire::new(6, DType::Bool), Wire::new(5, DType::Int), Wire::new(3, DType::Int)],
        ).unwrap(),
        // cond2: if diff2 < 0 then 0 else diff2 -> idx 9
        Term::function(
            IType::Cond,
            [Wire::new(9, DType::Int)],
            vec![Wire::new(7, DType::Bool), Wire::new(5, DType::Int), Wire::new(4, DType::Int)],
        ).unwrap(),
        // add relu1 + relu2 -> idx 10
        Term::function(
            IType::Add,
            [Wire::new(10, DType::Int)],
            vec![Wire::new(8, DType::Int); 2],
        ).unwrap(),
    ];

    // buchi condition: x0 = x1 ∨ x0 = x2
    let buchi: Vec<Term<DType, IType>> = vec![
        Term::function(
            IType::Eq,
            [Wire::new(3, DType::Bool)],
            vec![Wire::new(0, DType::Int); 2],
        ).unwrap(),
        Term::function(
            IType::Eq,
            [Wire::new(4, DType::Bool)],
            vec![Wire::new(0, DType::Int), Wire::new(2, DType::Int)],
        ).unwrap(),
        Term::function(
            IType::Or,
            [Wire::new(5, DType::Bool)],
            vec![Wire::new(3, DType::Bool); 2],
        ).unwrap(),
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

// cargo run --bin parse --package smv smv/tests/counter.smv --dump html --open
fn main() {
    let args = Cli::parse();
    let input = std::fs::read_to_string(&args.spec).expect("Cannot read the input file");

    let module = parse_smv(&input).unwrap();
    let lean = to_lean_from_module(&module).unwrap();
    let (invariant, variant, buchi) = build_obligations();
    let obs = module.obs();
    let latched = obs.latched();
    let mut wires_map: Vec<(String, usize, DType)> = Vec::new();
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for wire in latched {
        let i = wire.id();
        let dtype = wire.dtype();
        if seen.insert(i) {
            wires_map.push((format!("x{}", i), i, *dtype));
        }
    }
    let var_count = wires_map.len().saturating_sub(module.extl().latched().len());

    let inv_out = invariant.last().and_then(|t| t.write().wires().next()).expect("invariant output");
    let var_out = variant.last().and_then(|t| t.write().wires().next()).expect("variant output");
    let buch_out = buchi.last().and_then(|t| t.write().wires().next()).expect("buchi output");
    
    let inv_s = render_terms_to_lean(&invariant, inv_out, &wires_map, var_count);
    let var_s = render_terms_to_lean(&variant, var_out, &wires_map, var_count);
    let buch_s = render_terms_to_lean(&buchi, buch_out, &wires_map, var_count);

    // Compute variable tuples for each obligation by listing used VAR indices.
    let buch_vars = collect_used_vars(&buchi, buch_out, var_count);
    let inv_vars = collect_used_vars(&invariant, inv_out, var_count);
    let var_vars = collect_used_vars(&variant, var_out, var_count);
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
}
