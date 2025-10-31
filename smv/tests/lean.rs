use std::fs;
use smv::lean_parser::parse_smv_to_lean_view;
use smv::lean::to_lean;

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
            init(y) := y0;
            init(z) := z0;
            next(x) := (x < y | x < z) ? x + 1 : 0;
            next(y) := y;
            next(z) := z;
    "#;

    let view = parse_smv_to_lean_view(input).unwrap();
    let lean = to_lean(&view).unwrap();

    let out_path = format!("{}/tests/lean.md", env!("CARGO_MANIFEST_DIR"));
    std::fs::create_dir_all(format!("{}/tests", env!("CARGO_MANIFEST_DIR"))).unwrap();
    fs::write(out_path, lean).unwrap();
}
