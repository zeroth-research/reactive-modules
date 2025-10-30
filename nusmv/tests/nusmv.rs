use base::{atom::Atom, module::Module, term::Term, wire::Wire};
use nusmv::{dtype::DType, itype::IType, nusmv::parse_nusmv};
use nusmv::debug_utils::{dump_debug, compare_debug};

fn build_manual_module() -> Module<DType, IType> {
    let x = Wire::one(0, DType::Int);
    let y = Wire::one(1, DType::Int);
    let z = Wire::one(2, DType::Int);
    let y0 = Wire::one(3, DType::Int);
    let z0 = Wire::one(4, DType::Int);

    let wait = Wire::union(&y0, &z0).unwrap();
    let ctrl = x.union(&y).unwrap().union(&z).unwrap();
    let read = ctrl.clone();

    // init terms: x' := 0; y' := y0; z' := z0  (writes must target next wires)
    let init_terms = vec![
        Term::new(
            IType::Assign(Box::new(IType::VarRef("x".to_string())), Box::new(IType::ConstInt(0))),
            x.twin(5).unwrap(),
            Wire::none(),
        ),
        Term::new(
            IType::Assign(Box::new(IType::VarRef("y".to_string())), Box::new(IType::VarRef("y0".to_string()))),
            y.twin(5).unwrap(),
            y0.clone(),
        ),
        Term::new(
            IType::Assign(Box::new(IType::VarRef("z".to_string())), Box::new(IType::VarRef("z0".to_string()))),
            z.twin(5).unwrap(),
            z0.clone(),
        ),
    ];

    // update terms: next(x) := (x < y | x < z) ? x + 1 : 0; next(y) := y; next(z) := z;
    let update_x_itype = IType::Cond(
        Box::new(IType::Or(
            Box::new(IType::Lt(Box::new(IType::VarRef("x".to_string())), Box::new(IType::VarRef("y".to_string())))),
            Box::new(IType::Lt(Box::new(IType::VarRef("x".to_string())), Box::new(IType::VarRef("z".to_string())))),
        )),
        Box::new(IType::Add(Box::new(IType::VarRef("x".to_string())), Box::new(IType::ConstInt(1)))),
        Box::new(IType::ConstInt(0)),
    );

    let update_terms = vec![
        Term::new(
            update_x_itype,
            x.twin(5).unwrap(),
            x.union(&y).unwrap().union(&z).unwrap(),
        ),
        Term::new(
            IType::VarRef("y".to_string()),
            y.twin(5).unwrap(),
            y.clone(),
        ),
        Term::new(
            IType::VarRef("z".to_string()),
            z.twin(5).unwrap(),
            z.clone(),
        ),
    ];
    let atom = Atom::new_unchecked(
        ctrl.twin(5).unwrap(),
        wait.twin(5).unwrap(),
        read,
        init_terms,
        update_terms,
    );

    let latched = x
        .union(&y)
        .unwrap()
        .union(&z)
        .unwrap()
        .union(&y0)
        .unwrap()
        .union(&z0)
        .unwrap();
    let next = latched.twin(5).unwrap();

    Module::with_atoms([latched, next], vec![atom]).unwrap()
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

    dump_debug("Parsed Module", &parsed_module);
    dump_debug("Manual Module", &manual_module);
    compare_debug("Parsed", &parsed_module, "Manual", &manual_module);

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