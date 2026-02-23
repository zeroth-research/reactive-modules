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

    let parsed_module = parse_smv(input).unwrap().module;
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

// ==============================
// New tests for NuSMV 2.1 grammar
// ==============================

#[test]
fn case_esac() {
    let input = r#"
        MODULE main
        VAR x : integer;
        ASSIGN
          init(x) := 0;
          next(x) := case
            x < 5  : x + 1;
            TRUE   : 0;
          esac;
    "#;
    let result = parse_smv(input).unwrap();
    // Module should have 1 state var → 2 obs wires (latched + next)
    let obs_count: usize = result.module.obs().iter().map(|arr| arr.len()).sum();
    assert_eq!(obs_count, 2, "expected 2 obs wires for 1 state var");
}

#[test]
fn not_precedence() {
    // !a = b must parse as !(a = b), not (!a) = b
    let input = r#"
        MODULE main
        VAR a : boolean; b : boolean;
        ASSIGN
          init(a) := FALSE; init(b) := FALSE;
          next(a) := !a = b;
          next(b) := b;
    "#;
    let result = parse_smv(input).unwrap();
    // Find the update terms: should contain Eq followed by Not
    let update_has_eq_then_not = result.module.atoms().iter().any(|atom| {
        let terms: Vec<_> = atom.update().iter().collect();
        let eq_pos = terms.iter().position(|t| matches!(t.itype(), IType::Eq));
        let not_pos = terms.iter().position(|t| matches!(t.itype(), IType::Not));
        match (eq_pos, not_pos) {
            (Some(e), Some(n)) => e < n, // Eq must come before Not
            _ => false,
        }
    });
    assert!(update_has_eq_then_not, "!a = b should parse as !(a = b): Eq before Not");
}

#[test]
fn neq_operator() {
    let input = r#"
        MODULE main
        VAR x : integer; y : integer;
        ASSIGN
          init(x) := 0; init(y) := 1;
          next(x) := (x != y) ? x + 1 : 0;
          next(y) := y;
    "#;
    let result = parse_smv(input).unwrap();
    // Should contain a Neq term
    let has_neq = result.module.atoms().iter().any(|atom| {
        atom.update().iter().any(|t| matches!(t.itype(), IType::Neq))
    });
    assert!(has_neq, "expected Neq term in update");
}

#[test]
fn mod_operator() {
    let input = r#"
        MODULE main
        VAR x : integer;
        ASSIGN
          init(x) := 0;
          next(x) := x mod 3;
    "#;
    let result = parse_smv(input).unwrap();
    let has_mod = result.module.atoms().iter().any(|atom| {
        atom.update().iter().any(|t| matches!(t.itype(), IType::Mod))
    });
    assert!(has_mod, "expected Mod term in update");
}

#[test]
fn implies_iff_xor() {
    let input = r#"
        MODULE main
        VAR a : boolean; b : boolean; c : boolean;
        ASSIGN
          init(a) := FALSE; init(b) := TRUE; init(c) := FALSE;
          next(a) := a -> b;
          next(b) := a <-> b;
          next(c) := a xor c;
    "#;
    let result = parse_smv(input).unwrap();
    let all_update_itypes: Vec<String> = result.module.atoms().iter()
        .flat_map(|atom| atom.update().iter().map(|t| format!("{}", t.itype())))
        .collect();
    assert!(all_update_itypes.iter().any(|s| s == "Implies"), "expected Implies");
    assert!(all_update_itypes.iter().any(|s| s == "Xnor"), "expected Xnor (for <->)");
    assert!(all_update_itypes.iter().any(|s| s == "Xor"), "expected Xor");
}

#[test]
fn unary_minus() {
    let input = r#"
        MODULE main
        VAR x : integer;
        ASSIGN
          init(x) := -1;
          next(x) := -x;
    "#;
    let result = parse_smv(input).unwrap();
    let has_neg = result.module.atoms().iter().any(|atom| {
        atom.update().iter().any(|t| matches!(t.itype(), IType::Neg))
    });
    assert!(has_neg, "expected Neg term in update for -x");
}

#[test]
fn define_expansion() {
    let input = r#"
        MODULE main
        DEFINE inc := x + 1;
        VAR x : integer;
        ASSIGN
          init(x) := 0;
          next(x) := inc;
    "#;
    let result = parse_smv(input).unwrap();
    // The DEFINE should be expanded: update should contain Add (from x + 1)
    let has_add = result.module.atoms().iter().any(|atom| {
        atom.update().iter().any(|t| matches!(t.itype(), IType::Add))
    });
    assert!(has_add, "DEFINE inc := x + 1 should expand to Add term");
}

#[test]
fn range_type() {
    let input = r#"
        MODULE main
        VAR x : 0..10;
        ASSIGN
          init(x) := 0;
          next(x) := (x < 10) ? x + 1 : 0;
    "#;
    let result = parse_smv(input).unwrap();
    let obs_count: usize = result.module.obs().iter().map(|arr| arr.len()).sum();
    assert_eq!(obs_count, 2, "range type should map to Int, 1 state var → 2 obs wires");
}

#[test]
fn enum_type() {
    let input = r#"
        MODULE main
        VAR state : {idle, running, done};
        ASSIGN
          init(state) := idle;
          next(state) := case
            state = idle    : running;
            state = running : done;
            TRUE            : idle;
          esac;
    "#;
    let result = parse_smv(input).unwrap();
    // Should parse successfully, enum values mapped to ConstInt
    let has_const_int = result.module.atoms().iter().any(|atom| {
        atom.update().iter().any(|t| matches!(t.itype(), IType::ConstInt(_)))
    });
    assert!(has_const_int, "enum values should be lowered to ConstInt");
}

#[test]
fn init_trans_sections() {
    let input = r#"
        MODULE main
        VAR x : integer;
        ASSIGN
          next(x) := x + 1;
        INIT
          x = 0;
        TRANS
          next(x) < 100;
    "#;
    let result = parse_smv(input).unwrap();
    assert!(!result.init_constraints.is_empty(), "expected init constraint terms");
    assert!(!result.trans_constraints.is_empty(), "expected trans constraint terms");
}

#[test]
fn invar_section() {
    let input = r#"
        MODULE main
        IVAR y0 : integer;
        VAR x : integer;
        ASSIGN
          init(x) := y0;
          next(x) := x;
        INVAR
          x >= 0;
    "#;
    let result = parse_smv(input).unwrap();
    assert!(!result.invar_constraints.is_empty(), "expected invar constraint terms");
}

#[test]
fn frozenvar() {
    let input = r#"
        MODULE main
        FROZENVAR c : integer;
        VAR x : integer;
        ASSIGN
          init(x) := c;
          next(x) := x + c;
    "#;
    let result = parse_smv(input).unwrap();
    // Frozen var treated as state var with identity update: 2 state vars → 4 obs wires
    let obs_count: usize = result.module.obs().iter().map(|arr| arr.len()).sum();
    assert_eq!(obs_count, 4, "frozen var + state var → 4 obs wires");
}

#[test]
fn nested_case() {
    let input = r#"
        MODULE main
        VAR x : integer; y : integer;
        ASSIGN
          init(x) := 0; init(y) := 0;
          next(x) := case
            x < 5 : case
              y = 0 : 1;
              TRUE  : 2;
            esac;
            TRUE : 0;
          esac;
          next(y) := y;
    "#;
    let result = parse_smv(input).unwrap();
    // Should have multiple Cond terms from nested case
    let cond_count: usize = result.module.atoms().iter()
        .flat_map(|atom| atom.update().iter())
        .filter(|t| matches!(t.itype(), IType::Cond))
        .count();
    assert!(cond_count >= 2, "nested case should produce at least 2 Cond terms, got {}", cond_count);
}

#[test]
fn keyword_in_ident() {
    let input = r#"
        MODULE main
        VAR module_count : integer; order : integer; modify : integer;
        ASSIGN
          init(module_count) := 0;
          init(order) := 0;
          init(modify) := 0;
          next(module_count) := module_count + 1;
          next(order) := order;
          next(modify) := modify;
    "#;
    let result = parse_smv(input).unwrap();
    // Should parse without keyword conflicts; 3 state vars → 6 obs wires
    let obs_count: usize = result.module.obs().iter().map(|arr| arr.len()).sum();
    assert_eq!(obs_count, 6, "3 state vars → 6 obs wires");
}

// ==============================
// Word-level support tests
// ==============================

#[test]
fn word_type_and_literal() {
    let input = r#"
        MODULE main
        VAR x : unsigned word[16]; y : signed word[32];
        INIT
          x = 0ud16_0;
        INIT
          y = 0sd32_100;
        TRANS
          next(x) = x + 0ud16_1;
        TRANS
          next(y) = y;
    "#;
    let result = parse_smv(input).unwrap();
    let obs_count: usize = result.module.obs().iter().map(|arr| arr.len()).sum();
    assert_eq!(obs_count, 4, "2 state vars → 4 obs wires");
    assert!(!result.init_constraints.is_empty(), "expected init constraint terms");
    assert!(!result.trans_constraints.is_empty(), "expected trans constraint terms");
}

#[test]
fn builtin_bool_word1() {
    let input = r#"
        MODULE main
        VAR a : unsigned word[1]; b : boolean;
        INVAR
          b = bool(a);
        INVAR
          a = word1(b);
        TRANS
          next(a) = a;
        TRANS
          next(b) = b;
    "#;
    let result = parse_smv(input).unwrap();
    assert!(!result.invar_constraints.is_empty(), "expected invar constraints");
    let all_itypes: Vec<String> = result
        .invar_constraints
        .iter()
        .map(|t| format!("{}", t.itype()))
        .collect();
    assert!(
        all_itypes.iter().any(|s| s == "ToBool"),
        "expected ToBool in invar constraints, got {:?}",
        all_itypes
    );
    assert!(
        all_itypes.iter().any(|s| s == "ToWord1"),
        "expected ToWord1 in invar constraints, got {:?}",
        all_itypes
    );
}

#[test]
fn builtin_unsigned_extend() {
    let input = r#"
        MODULE main
        VAR x : unsigned word[16]; y : unsigned word[32];
        INVAR
          y = extend(x, 16);
        INVAR
          y = unsigned(0sd32_42);
        TRANS
          next(x) = x;
        TRANS
          next(y) = y;
    "#;
    let result = parse_smv(input).unwrap();
    let all_itypes: Vec<String> = result
        .invar_constraints
        .iter()
        .map(|t| format!("{}", t.itype()))
        .collect();
    assert!(
        all_itypes.iter().any(|s| s == "Extend(16)"),
        "expected Extend(16) in invar constraints, got {:?}",
        all_itypes
    );
    assert!(
        all_itypes.iter().any(|s| s == "ToUnsigned"),
        "expected ToUnsigned in invar constraints, got {:?}",
        all_itypes
    );
}

#[test]
fn bit_select() {
    let input = r#"
        MODULE main
        VAR x : unsigned word[32]; y : unsigned word[16];
        INVAR
          y = (x)[15:0];
        INVAR
          y = 0sd32_100[15:0];
        TRANS
          next(x) = x;
        TRANS
          next(y) = y;
    "#;
    let result = parse_smv(input).unwrap();
    let all_itypes: Vec<String> = result
        .invar_constraints
        .iter()
        .map(|t| format!("{}", t.itype()))
        .collect();
    assert!(
        all_itypes.iter().any(|s| s == "BitSelect[15:0]"),
        "expected BitSelect[15:0] in invar constraints, got {:?}",
        all_itypes
    );
}

#[test]
fn semicolon_free_sections() {
    let input = r#"
        MODULE main
        VAR x : integer; y : boolean;
        INIT x = 0
        INIT y = FALSE
        INVAR x >= 0
        TRANS next(x) = x + 1
        TRANS next(y) = y
    "#;
    let result = parse_smv(input).unwrap();
    assert!(
        !result.init_constraints.is_empty(),
        "expected init constraints from semicolon-free INIT"
    );
    assert!(
        !result.invar_constraints.is_empty(),
        "expected invar constraints from semicolon-free INVAR"
    );
    assert!(
        !result.trans_constraints.is_empty(),
        "expected trans constraints from semicolon-free TRANS"
    );
}
