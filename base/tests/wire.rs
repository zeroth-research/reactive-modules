use base::Wire;
use base::wire::Interface;

#[test]
fn can_instantiate_wire() {
    let x = Interface::sequence([Wire::new("real")]).unwrap();
    let y = Interface::sequence((4..14).map(|_| Wire::new("int"))).unwrap();

    let w = Interface::try_from_iter(x.into_iter().chain(y));
    assert!(w.is_ok())
}

#[test]
fn union_out_of_order() {
    let x = Interface::sequence((0..10).map(|_| Wire::new("real"))).unwrap();
    let y = Interface::sequence((4..10).map(|_| Wire::new("real"))).unwrap();

    let w = Interface::try_from_iter([y, x].into_iter().flatten());
    assert!(w.is_ok());
}
