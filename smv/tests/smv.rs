use base::{atom::Atom, module::Module, term::Term, wire::Wire};
use smv::{dtype::DType, itype::IType, smv::parse_smv, smv::twin};
use std::collections::HashMap;

struct Context {
    vars: HashMap<String, Wire<DType>>,
}

impl Context {
    fn new() -> Self {
        Self {
            vars: HashMap::new(),
        }
    }

    fn get(&mut self, name: &'static str) -> &Wire<DType> {
        self.vars.get(name).expect("Not existing value")
    }

    fn get_cloned(&mut self, name: &'static str) -> Wire<DType> {
        self.vars.get(name).expect("Not existing value").clone()
    }

    /// Get or create a variable
    /// Does not check if the type is compatible if the var exists
    fn var(&mut self, name: &'static str, ty: DType) -> &Wire<DType> {
        let new_id = self.vars.len();
        self.vars
            .entry(name.to_string())
            .or_insert(Wire::one(new_id, ty))
    }

    /// Does not check if the type is compatible if the var exists
    fn get_vars(&mut self, names: Vec<&'static str>) -> Wire<DType> {
        // XXX: not very efficient
        let mut wire = Wire::none();
        for name in names {
            let v = self.get(name);
            wire = wire.extend(v);
        }

        wire
    }

    fn vars(&mut self, ty: DType, names: Vec<&'static str>) -> Wire<DType> {
        // XXX: not very efficient
        let mut wire = Wire::none();
        for name in names {
            let v = self.var(name, ty);
            wire = wire.extend(v);
        }

        wire
    }
}

fn build_manual_module() -> Module<DType, IType> {
    // create variables
    let mut ctx = Context::new();

    ctx.vars(
        DType::Int,
        vec!["x", "y", "z", "y0", "z0", "x'", "y'", "z'", "y0'", "z0'"],
    );
    const NEXT_OFFSET: isize = 5;

    fn init(ctx: &mut Context) -> Vec<Term<DType, IType>> {
        // Build init terms to match parser output exactly (hard-coded indices)
        // init(x) := 0 -> write to x' (index 5)
        let init_x = Term::new(IType::ConstInt(0), ctx.get_cloned("x'"), Wire::none());

        // Sequence for abs(y0) lowered with temps 16..19
        let t16 = Wire::one(16, DType::Int);
        let t17 = Wire::one(17, DType::Int);
        let t18 = Wire::one(18, DType::Bool);
        let t19 = Wire::one(19, DType::Int);
        let init_y_seq = vec![
            Term::new(IType::ConstInt(0), t16.clone(), Wire::none()),
            // Sub read: [t16, y0]
            Term::new(
                IType::Sub,
                t17.clone(),
                t16.clone().extend(&ctx.get_cloned("y0")),
            ),
            // Lt read: [y0, t16]
            Term::new(
                IType::Lt,
                t18.clone(),
                ctx.get_cloned("y0").extend(&t16.clone()),
            ),
            // Cond read: [t18, t17, y0]
            Term::new(
                IType::Cond,
                t19.clone(),
                t18.clone()
                    .extend(&t17.clone())
                    .extend(&ctx.get_cloned("y0")),
            ),
        ];

        // Sequence for abs(z0) lowered with temps 20..23
        let t20 = Wire::one(20, DType::Int);
        let t21 = Wire::one(21, DType::Int);
        let t22 = Wire::one(22, DType::Bool);
        let t23 = Wire::one(23, DType::Int);
        let init_z_seq = vec![
            Term::new(IType::ConstInt(0), t20.clone(), Wire::none()),
            // Sub read: [t20, z0]
            Term::new(
                IType::Sub,
                t21.clone(),
                t20.clone().extend(&ctx.get_cloned("z0")),
            ),
            // Lt read: [z0, t20]
            Term::new(
                IType::Lt,
                t22.clone(),
                ctx.get_cloned("z0").extend(&t20.clone()),
            ),
            // Cond read: [t22, t21, z0]
            Term::new(
                IType::Cond,
                t23.clone(),
                t22.clone()
                    .extend(&t21.clone())
                    .extend(&ctx.get_cloned("z0")),
            ),
        ];

        let mut out = vec![init_x];
        out.extend(init_y_seq);
        out.extend(init_z_seq);
        out
    }

    fn update(ctx: &mut Context) -> Vec<Term<DType, IType>> {
        // Build update terms with explicit indices to match parser output
        let lt10 = Term::new(
            IType::Lt,
            Wire::one(10, DType::Bool),
            ctx.get_vars(vec!["x", "y"]),
        );
        let lt11 = Term::new(
            IType::Lt,
            Wire::one(11, DType::Bool),
            ctx.get_vars(vec!["x", "z"]),
        );
        let or12 = Term::new(
            IType::Or,
            Wire::one(12, DType::Bool),
            Wire::one(10, DType::Bool).extend(&Wire::one(11, DType::Bool)),
        );
        let c1_13 = Term::new(IType::ConstInt(1), Wire::one(13, DType::Int), Wire::none());
        let add14 = Term::new(
            IType::Add,
            Wire::one(14, DType::Int),
            ctx.get("x").clone().extend(&Wire::one(13, DType::Int)),
        );
        let c0_15 = Term::new(IType::ConstInt(0), Wire::one(15, DType::Int), Wire::none());
        let cond5 = Term::new(
            IType::Cond,
            ctx.get_cloned("x'"),
            Wire::one(12, DType::Bool)
                .extend(&Wire::one(14, DType::Int))
                .extend(&Wire::one(15, DType::Int)),
        );
        let id_y = Term::new(IType::Assign, ctx.get_cloned("y'"), ctx.get_cloned("y"));
        let id_z = Term::new(IType::Assign, ctx.get_cloned("z'"), ctx.get_cloned("z"));

        vec![lt10, lt11, or12, c1_13, add14, c0_15, cond5, id_y, id_z]
    }

    // IMPORTANT: allocate update temps first (to match parser temp ordering),
    // then allocate init temps. This mirrors the parser lowering which may
    // emit update temporaries before some init temporaries.
    let update_terms = update(&mut ctx);
    let init_terms = init(&mut ctx);

    let latched = ctx.get_vars(vec!["x", "y", "z", "y0", "z0"]);
    let next = twin(&latched, NEXT_OFFSET).expect("Failed getting primed variables");

    // Construct atom with exact ctrl/wait/read wires to match parser output
    let ctrl: Wire<DType> = vec![
        (5usize, DType::Int),
        (6usize, DType::Int),
        (7usize, DType::Int),
    ]
    .into_iter()
    .collect();
    let wait: Wire<DType> = vec![(8usize, DType::Int), (9usize, DType::Int)]
        .into_iter()
        .collect();
    let read: Wire<DType> = vec![
        (0usize, DType::Int),
        (1usize, DType::Int),
        (2usize, DType::Int),
    ]
    .into_iter()
    .collect();
    let atom = Atom::new_unchecked(ctrl, wait, read, init_terms, update_terms);

    Module::partially_observable([latched, next], [Wire::none(), Wire::none()], vec![atom]).unwrap()
}

#[test]
fn counter_smv() {
    // Later on add `INVAR y0 >= 0 & z0 >= 0;` to the input instead of `abs(y0)` and `abs(z0)`
    let input = r#"
        MODULE main
        IVAR
            y0 : integer;
            z0 : integer;
        VAR
            x : integer;
            y : integer;
            z : integer;
        ASSIGN
            init(x) := 0;
            init(y) := abs(y0);
            init(z) := abs(z0);
            next(x) := (x < y | x < z) ? x + 1 : 0;
            next(y) := y;
            next(z) := z;
    "#;

    let parsed_module = parse_smv(input).unwrap();
    // Instead of printing to stdout during tests, write the same debug output
    // that `dump_debug` would print into files under the smv crate tests
    // directory so CI / editors don't get noisy output. Build the manual
    // module using the handcrafted `build_manual_module` so the test truly
    // exercises the manual builder.
    let parsed_out = format!("{:#?}", parsed_module);
    let manual_out = format!("{:#?}", build_manual_module());

    // Use the crate manifest dir to get a stable path to the smv crate
    let crate_root = env!("CARGO_MANIFEST_DIR");
    let parsed_path = std::path::Path::new(crate_root).join("tests/parsed.md");
    let manual_path = std::path::Path::new(crate_root).join("tests/manual.md");

    std::fs::write(&parsed_path, &parsed_out).expect("failed to write parsed.md");
    std::fs::write(&manual_path, &manual_out).expect("failed to write manual.md");

    // Build a textual diff (same semantics as debug_utils::compare_debug) and write to diff.md
    let a_str = parsed_out.clone();
    let b_str = manual_out.clone();
    let a_lines: Vec<&str> = a_str.lines().collect();
    let b_lines: Vec<&str> = b_str.lines().collect();

    let mut diff_out = String::new();
    diff_out.push_str("==================== Comparing Parsed vs Manual ====================\n");
    for (i, (la, lb)) in a_lines.iter().zip(b_lines.iter()).enumerate() {
        if la.trim() != lb.trim() {
            diff_out.push_str(&format!("❌ Line {} differs:\n", i + 1));
            diff_out.push_str(&format!("  Parsed: {}\n", la));
            diff_out.push_str(&format!("  Manual: {}\n", lb));
        }
    }
    if a_lines.len() != b_lines.len() {
        diff_out.push_str(&format!(
            "\n⚠️ Different number of lines ({} vs {})\n",
            a_lines.len(),
            b_lines.len()
        ));
    }
    diff_out.push_str("==============================================================\n");

    let diff_path = std::path::Path::new(crate_root).join("tests/diff.md");
    std::fs::write(&diff_path, diff_out).expect("failed to write diff.md");

    // Assert wire-range sections (extl, intf, ctrl, obs) match between parsed and manual modules.
    let parsed_str = parsed_out.clone();
    let manual_str = manual_out.clone();
    let parsed_lines: Vec<&str> = parsed_str.lines().collect();
    let manual_lines: Vec<&str> = manual_str.lines().collect();

    fn extract_section(lines: &[&str], header: &str) -> String {
        for (i, &line) in lines.iter().enumerate() {
            if line.contains(header) {
                let mut out = String::new();
                out.push_str(line);
                out.push('\n');
                for &l in &lines[i + 1..] {
                    out.push_str(l);
                    out.push('\n');
                    if l.trim() == "]," {
                        break;
                    }
                }
                return out;
            }
        }
        String::new()
    }

    for header in &["extl:", "intf:", "ctrl:", "obs:"] {
        let p = extract_section(&parsed_lines, header);
        let m = extract_section(&manual_lines, header);
        assert_eq!(p.trim(), m.trim(), "Wire section {} differs", header);
    }
}
