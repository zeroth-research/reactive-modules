use base::wire::Interface;

#[test]
fn can_instantiate_wire() {
    let x = Interface::single(0, "real");
    let y = Interface::sequence((4..14).map(|i| (i, "int"))).unwrap();

    let w = Interface::try_from_iter(x.into_iter().chain(y));
    assert!(w.is_ok())
}

#[test]
fn cannot_join_incompatible_wires() {
    let x = Interface::sequence((0..10).map(|i| (i, "real"))).unwrap();
    let y = Interface::sequence((4..10).map(|i| (i, "bool"))).unwrap();

    let w = Interface::try_from_iter([x, y].into_iter().flatten());
    assert!(w.is_err());
}

#[test]
fn union_out_of_order() {
    let x = Interface::sequence((0..10).map(|i| (i, "real"))).unwrap();
    let y = Interface::sequence((4..10).map(|i| (i, "real"))).unwrap();

    let w = Interface::try_from_iter([y, x].into_iter().flatten());
    assert!(w.is_ok());
}
