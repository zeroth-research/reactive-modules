use std::collections::HashMap;
use base::{atom::Atom, module::Module, term::Term, wire::Wire};
use nusmv::{dtype::DType, itype::IType, nusmv::parse_nusmv};

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
    fn tmp_var(&mut self, ty: DType) -> &Wire<DType> {
        let new_id = self.vars.len();
        self.vars
            .entry(format!("__c_{}", new_id))
            .or_insert(Wire::one(new_id, ty))
    }

    fn get_vars(&mut self, names: Vec<&'static str>) -> Wire<DType> {
        // XXX: not very efficient
        let mut wire = Wire::none();
        for name in names {
            let v = self.get(name);
            wire = wire.union(v).unwrap();
        }

        wire
    }

    fn vars(&mut self, ty: DType, names: Vec<&'static str>) -> Wire<DType> {
        // XXX: not very efficient
        let mut wire = Wire::none();
        for name in names {
            let v = self.var(name, ty);
            wire = wire.union(v).unwrap();
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
        let init_x = Term::new(
            IType::ConstInt(0),
            ctx.get_cloned("x'"),
            Wire::none(),
        );
        let init_y =
            Term::new(IType::Assign, ctx.get_cloned("y'"), ctx.get_cloned("y0'"));
        let init_z =
            Term::new(IType::Assign, ctx.get_cloned("z'"), ctx.get_cloned("z0'"));

        vec![init_x, init_y, init_z]
    }

    fn update(ctx: &mut Context) -> Vec<Term<DType, IType>> {
        // wire10 = x < y
        let reads = ctx.get_vars(vec!["x", "y"]);
        let wire10 = ctx.tmp_var(DType::Bool).clone();
        let xlty = Term::new(IType::Lt, wire10.clone(), reads);

        // wire11 = x < z
        let reads = ctx.get_vars(vec!["x", "z"]);
        let wire11 = ctx.tmp_var(DType::Bool).clone();
        let xltz = Term::new(IType::Lt, wire11.clone(), reads);

        // wire12 = wire10 || wire11
        let wire12 = ctx.tmp_var(DType::Bool).clone();
        let reads = wire10.union(&wire11).unwrap();
        let or = Term::new(IType::Or, wire12.clone(), reads);

        // one
        let const1 = ctx.tmp_var(DType::Int).clone();
        let term1 = Term::new(
            IType::ConstInt(1),
            const1.clone(),
            Wire::none(),
        );

        // wire14 = vars[0] + const1
        let wire14 = ctx.tmp_var(DType::Int).clone();
        let reads = ctx.get("x").union(&const1).unwrap();
        let sum = Term::new(IType::Add, wire14.clone(), reads);

        // zero
        let const0 = ctx.tmp_var(DType::Int).clone();
        let term0 = Term::new(
            IType::ConstInt(0),
            const0.clone(),
            Wire::none(),
        );

        // wire5 = ite(wire12, wire15, const0)
        let reads = wire12.union(&wire14).unwrap().union(&const0).unwrap();
        let ite = Term::new(IType::Cond, ctx.get_cloned("x'"), reads);

        // y' := y
        let id_y = Term::new(IType::Assign, ctx.get_cloned("y'"), ctx.get_cloned("y"));
        let id_z = Term::new(IType::Assign, ctx.get_cloned("z'"), ctx.get_cloned("z"));

        vec![xlty, xltz, or, term0, term1, sum, ite, id_y, id_z]
    }

    let init_terms = init(&mut ctx);
    let update_terms = update(&mut ctx);

    let latched = ctx.get_vars(vec!["x", "y", "z", "y0", "z0"]);
    let next = latched
        .twin(NEXT_OFFSET)
        .expect("Failed getting primed variables");

    let atom =
        Atom::with_module_wire(&[latched.clone(), next.clone()], init_terms, update_terms)
            .expect("failed creating atom");

    Module::with_atoms([latched, next], vec![atom]).expect("Failed building module")
}

#[test]
fn counter_nusmv() {
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
            init(y) := y0;
            init(z) := z0;
            next(x) := (x < y | x < z) ? x + 1 : 0;
            next(y) := y;
            next(z) := z;
    "#;

    let parsed_module = parse_nusmv(input).unwrap();
    let manual_module = build_manual_module();

    // Instead of printing to stdout during tests, write the same debug output
    // that `dump_debug` would print into files under the nusmv crate tests
    // directory so CI / editors don't get noisy output.
    let parsed_out = format!("{:#?}", parsed_module);
    let manual_out = format!("{:#?}", manual_module);

    // Use the crate manifest dir to get a stable path to the nusmv crate
    let crate_root = env!("CARGO_MANIFEST_DIR");
    let parsed_path = std::path::Path::new(crate_root).join("tests/parsed.md");
    let manual_path = std::path::Path::new(crate_root).join("tests/manual.md");

    std::fs::write(&parsed_path, parsed_out).expect("failed to write parsed.md");
    std::fs::write(&manual_path, manual_out).expect("failed to write manual.md");

    // Build a textual diff (same semantics as debug_utils::compare_debug) and write to diff.md
    let a_str = format!("{:#?}", parsed_module);
    let b_str = format!("{:#?}", manual_module);
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
        diff_out.push_str(&format!("\n⚠️ Different number of lines ({} vs {})\n", a_lines.len(), b_lines.len()));
    }
    diff_out.push_str("==============================================================\n");

    let diff_path = std::path::Path::new(crate_root).join("tests/diff.md");
    std::fs::write(&diff_path, diff_out).expect("failed to write diff.md");

    // Assert wire-range sections (extl, intf, ctrl, obs) match between parsed and manual modules.
    let parsed_str = format!("{:#?}", parsed_module);
    let manual_str = format!("{:#?}", manual_module);
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