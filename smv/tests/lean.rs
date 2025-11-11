use base::term::{Term, TermWire};
use base::wire::Wire;
use smv::lean::{collect_used_vars, render_terms_to_lean, to_lean_from_module};
use smv::smv::parse_smv;
use smv::{dtype::DType, itype::IType};
use std::fs;

// Build the obligation Term vectors (Büchi, invariant, variant) used by the
// tests. Kept as a separate function so the test body stays concise and the
// obligation construction can be reused or unit-tested independently.
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

#[test]
fn write_lean_example() {
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

    // Create a Module by parsing the input, then render it to Lean.
    let module = parse_smv(input).unwrap();
    let lean = to_lean_from_module(&module).unwrap();
    // Build obligations as Terms (same structure as obligations/tests/obligations.rs).
    // Construction has been moved into `build_obligations()` above for reuse.
    let (invariant, variant, buchi) = build_obligations();

    // Reconstruct the module wires mapping (name,index,dtype) like the renderer does
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

    let out_path = format!("{}/tests/lean.md", env!("CARGO_MANIFEST_DIR"));
    std::fs::create_dir_all(format!("{}/tests", env!("CARGO_MANIFEST_DIR"))).unwrap();
    fs::write(out_path, full).unwrap();
}
