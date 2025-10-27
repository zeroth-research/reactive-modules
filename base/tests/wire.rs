use base::wire::Wire;

#[test]
fn can_instantiate_rugged_wire() {
    let x = Wire::scalar(0, "real");
    let y = Wire::vector(4, "int", 10);

    let w = Wire::union(&x, &y);
    assert!(w.is_ok())
}

#[test]
fn cannot_join_incompatible_wires() {
    let x = Wire::vector(0, "real", 10);
    let y = Wire::vector(4, "int", 10);

    let w = Wire::union(&x, &y);
    assert!(w.is_err());
}

#[test]
fn union_out_of_order() {
    let x = Wire::vector(0, "real", 4);
    let y = Wire::vector(4, "real", 4);

    let w = Wire::union(&y, &x);
    assert!(w.is_ok());
}

#[test]
fn intersect_two_consecutive_wires() {
    let x = Wire::vector(0, "real", 10);
    let y = Wire::vector(20, "real", 10);
    let z = Wire::union(&x, &y).unwrap();
    let w = Wire::vector(5, "real", 20);

    let a = Wire::intersection(&z, &w);
    assert!(a.is_ok());
    assert_eq!(a.unwrap().size(), 10);
}

#[test]
fn intersect_three_consecutive_wires() {
    let x = Wire::vector(0, "real", 10);
    let y = Wire::vector(20, "real", 10);
    let z = Wire::vector(40, "real", 10);
    let x = Wire::union(&x, &y).unwrap();
    let x = Wire::union(&x, &z).unwrap();
    let w = Wire::vector(5, "real", 40);

    let a = Wire::intersection(&x, &w).unwrap();
    assert_eq!(a.size(), 20);
}

#[test]
fn intersect_between_two_consecutive_wires() {
    let x = Wire::union(&Wire::vector(0, "real", 10), &Wire::vector(20, "real", 10)).unwrap();
    let y = Wire::vector(10, "real", 10);

    let z = Wire::intersection(&x, &y).unwrap();
    assert_eq!(z.size(), 0);
}

#[test]
fn intersect_second_of_two_arrays() {
    let x = Wire::union(&Wire::vector(0, "real", 10), &Wire::vector(20, "real", 10)).unwrap();
    let y = Wire::vector(10, "real", 15);

    let z = Wire::intersection(&x, &y).unwrap();
    assert_eq!(z.size(), 5);
}

#[test]
fn intersect_first_of_two_arrays() {
    let x = Wire::union(&Wire::vector(0, "real", 10), &Wire::vector(20, "real", 10)).unwrap();
    let y = Wire::vector(5, "real", 15);

    let z = Wire::intersection(&x, &y).unwrap();
    assert_eq!(z.size(), 5);
}

#[test]
fn strongly_intersect_first_of_two_arrays() {
    let x = Wire::union(&Wire::vector(0, "real", 10), &Wire::vector(20, "real", 10)).unwrap();
    let y = Wire::vector(5, "real", 2);

    let z = Wire::intersection(&x, &y).unwrap();
    assert_eq!(z.size(), 2);
}

#[test]
fn strongly_intersect_second_of_two_arrays() {
    let x = Wire::union(&Wire::vector(0, "real", 10), &Wire::vector(20, "real", 10)).unwrap();
    let y = Wire::vector(21, "real", 2);

    let z = Wire::intersection(&x, &y).unwrap();
    assert_eq!(z.size(), 2);
}

#[test]
fn cannot_intersect_incompatible_wires() {
    let x = Wire::union(&Wire::vector(0, "real", 10), &Wire::vector(20, "int", 10)).unwrap();
    let y = Wire::vector(0, "real", 30);

    let z = Wire::intersection(&x, &y);
    assert!(z.is_err());
}

#[test]
fn difference_halfway_left() {
    let x = Wire::vector(10, "real", 10);
    let y = Wire::vector(0, "real", 15);

    let z = x.difference(&y).unwrap();
    assert_eq!(z.size(), 5);
}

#[test]
fn difference_halfway_right() {
    let x = Wire::vector(10, "real", 10);
    let y = Wire::vector(15, "real", 15);

    let z = x.difference(&y).unwrap();
    assert_eq!(z.size(), 5);
}

#[test]
fn difference_midway() {
    let x = Wire::vector(10, "real", 10);
    let y = Wire::vector(12, "real", 2);

    let z = x.difference(&y).unwrap();
    assert_eq!(z.size(), 8);
}

#[test]
fn difference_all_the_way() {
    let x = Wire::vector(10, "real", 10);
    let y = Wire::vector(0, "real", 30);

    let z = x.difference(&y).unwrap();
    assert_eq!(z.size(), 0);
}

#[test]
fn difference_across_two() {
    let x = Wire::vector(0, "real", 10);
    let y = Wire::vector(20, "real", 10);
    let z = Wire::vector(5, "real", 20);

    let w = x.union(&y).unwrap().difference(&z).unwrap();
    assert_eq!(w.size(), 10);
}

#[test]
fn difference_between_two() {
    let x = Wire::vector(0, "real", 10);
    let y = Wire::vector(20, "real", 10);
    let z = Wire::vector(11, "real", 5);

    let mut w = x.union(&y).unwrap();
    w = w.difference(&z).unwrap();
    assert_eq!(w.size(), 20);
}
