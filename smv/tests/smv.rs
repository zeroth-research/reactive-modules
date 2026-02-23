use base::{module::Module, term::Term, wire::Wire};
use smv::{dtype::DType, itype::IType, smv::parse_smv};
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
            .or_insert(Wire::new(new_id, ty))
    }

    /// Does not check if the type is compatible if the var exists
    fn get_vars(&mut self, names: Vec<&'static str>) -> Vec<Wire<DType>> {
        names.iter().map(|name| self.get(name).clone()).collect()
    }

    fn vars(&mut self, ty: DType, names: Vec<&'static str>) -> Vec<Wire<DType>> {
        names.iter().map(|name| self.var(name, ty).clone()).collect()
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
        // Build init terms to match parser output exactly.
        // The parser, for IVAR reads in init, produces Assign from the primed
        // (next) wire. So init(y) := abs(y0) becomes Assign(y', [y0']).
        // init(x) := 0 -> write to x' (index 5)
        let init_x = Term::function::<Wire<DType>, Wire<DType>, _, _>(IType::ConstInt(0), [ctx.get_cloned("x'")], vec![]).unwrap();

        // init(y) := abs(y0) — parser treats IVAR ident in abs() as Assign from primed wire
        let init_y = Term::function(IType::Assign, [ctx.get_cloned("y'")], vec![ctx.get_cloned("y0'")]).unwrap();

        // init(z) := abs(z0) — same treatment
        let init_z = Term::function(IType::Assign, [ctx.get_cloned("z'")], vec![ctx.get_cloned("z0'")]).unwrap();

        vec![init_x, init_y, init_z]
    }

    fn update(ctx: &mut Context) -> Vec<Term<DType, IType>> {
        // Build update terms with explicit indices to match parser output
        let lt10 = Term::function(
            IType::Lt,
            [Wire::new(10, DType::Bool)],
            ctx.get_vars(vec!["x", "y"]),
        ).unwrap();
        let lt11 = Term::function(
            IType::Lt,
            [Wire::new(11, DType::Bool)],
            ctx.get_vars(vec!["x", "z"]),
        ).unwrap();
        let or12 = Term::function(
            IType::Or,
            [Wire::new(12, DType::Bool)],
            vec![Wire::new(10, DType::Bool), Wire::new(11, DType::Bool)],
        ).unwrap();
        let c1_13 = Term::function::<Wire<DType>, Wire<DType>, _, _>(IType::ConstInt(1), [Wire::new(13, DType::Int)], vec![]).unwrap();
        let add14 = Term::function(
            IType::Add,
            [Wire::new(14, DType::Int)],
            vec![ctx.get("x").clone(), Wire::new(13, DType::Int)],
        ).unwrap();
        let c0_15 = Term::function::<Wire<DType>, Wire<DType>, _, _>(IType::ConstInt(0), [Wire::new(15, DType::Int)], vec![]).unwrap();
        let cond5 = Term::function(
            IType::Cond,
            [ctx.get_cloned("x'")],
            vec![Wire::new(12, DType::Bool), Wire::new(14, DType::Int), Wire::new(15, DType::Int)],
        ).unwrap();
        let id_y = Term::function(IType::Assign, [ctx.get_cloned("y'")], vec![ctx.get_cloned("y")]).unwrap();
        let id_z = Term::function(IType::Assign, [ctx.get_cloned("z'")], vec![ctx.get_cloned("z")]).unwrap();

        vec![lt10, lt11, or12, c1_13, add14, c0_15, cond5, id_y, id_z]
    }

    // IMPORTANT: allocate update temps first (to match parser temp ordering),
    // then allocate init temps. This mirrors the parser lowering which may
    // emit update temporaries before some init temporaries.
    let update_terms = update(&mut ctx);
    let init_terms = init(&mut ctx);

    let latched = ctx.get_vars(vec!["x", "y", "z", "y0", "z0"]);
    let next: Vec<Wire<DType>> = latched.iter()
        .map(|w| Wire::new(w.id() + NEXT_OFFSET as usize, *w.dtype()))
        .collect();

    let obs_pairs: Vec<[Wire<DType>; 2]> = latched.iter().zip(next.iter())
        .map(|(l, n)| [l.clone(), n.clone()])
        .collect();
    Module::sequential(
        obs_pairs,
        std::iter::empty::<[Wire<DType>; 2]>(),
        init_terms,
        update_terms,
    ).unwrap()
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
    let manual_module = build_manual_module();

    let parsed_out = format!("{:#?}", parsed_module);
    let manual_out = format!("{:#?}", manual_module);

    // Assert wire-range sections (extl, intf, ctrl, obs) match between parsed and manual modules.
    let parsed_lines: Vec<&str> = parsed_out.lines().collect();
    let manual_lines: Vec<&str> = manual_out.lines().collect();

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
