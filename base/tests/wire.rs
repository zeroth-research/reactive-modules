use base::wire::Wire;

#[test]
fn can_instantiate_rugged_wire() {
    let x = Wire::one(0, "real");
    let y = Wire::many(4, "int", 10);

    let w = Wire::try_from_iter([x, y].into_iter().flatten());
    assert!(w.is_ok())
}

#[test]
fn cannot_join_incompatible_wires() {
    let x = Wire::many(0, "real", 10);
    let y = Wire::many(4, "int", 10);

    let w = Wire::try_from_iter([x, y].into_iter().flatten());
    assert!(w.is_err());
}

#[test]
fn union_out_of_order() {
    let x = Wire::many(0, "real", 4);
    let y = Wire::many(4, "real", 4);

    let w = Wire::try_from_iter([x, y].into_iter().flatten());
    assert!(w.is_ok());
}
