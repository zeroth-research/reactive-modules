use base::var::{X, X_ref, d, d_ref};
use base::{Term, Var, Wire};
use theory::lra::{LRA, Sort};

#[test]
fn variables_are_globally_ordered_by_creation() {
    let x = Var::new("t");
    let y = Var::new("t");
    let z = Var::new("t");

    // the latched wire's id increases with creation
    assert!(x < y && y < z);
    assert!(x <= x && x == x);

    // sorting recovers the creation order
    let mut vars = [z, x, y];
    vars.sort();
    assert_eq!(vars, [x, y, z]);
}

fn borrow_copy_and_compare(x: &Var<&str>, nxt: Wire<&str>, der: Wire<&str>) {
    assert_eq!(d(x), nxt);
    assert_eq!(X(x), der);
}

#[test]
fn variable_casts_to_its_wires() {
    // variables are Copy: owned bindings, ampersand-free use everywhere
    let x = Var::new("type");

    let a: Wire<_> = *x; // the latched wire
    let b: Wire<_> = d(x); //     its derivative
    let c: Wire<_> = X(x); //   the next wire

    // d/nxt copy, never clone and never consume: the variable stays
    // usable and the views are stable
    assert_eq!(d(x), b);
    assert_eq!(X(x), c);

    // three distinct wires; the latched and next carry the value sort,
    // the derivative carries its tangent (the identity for string sorts)
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);

    // the variable dereferences to its latched wire: `Wire`'s methods
    // apply to it directly, and `&x` coerces to `&Wire<_>`
    assert_eq!(x.id(), a.id());
    let borrowed: &Wire<_> = &x;
    assert_eq!(*borrowed, a);

    // X and d take any borrow of the variable: by reference as well
    borrow_copy_and_compare(&x, b, c);

    // the _ref alternatives borrow all the way through: reference in,
    // reference out, no clone
    let da: &Wire<_> = &x;
    let db: &Wire<_> = d_ref(&x);
    let cb: &Wire<_> = X_ref(&x);
    assert_eq!(da, &a);
    assert_eq!(db, &b);
    assert_eq!(cb, &c);
}

#[test]
fn variables_pass_into_term_constructors() {
    use crate::Term;
    use theory::lra::{LRA, Sort};

    let x = Var::new(Sort::real([1, 1]));

    // a variable stands for its latched wire in a term position
    let t = Term::constant(LRA::AnyReal([1, 1]), [x]).unwrap();
    assert_eq!(t.write().iter().next().map(Wire::id), Some(x.id()));

    // and its views pass as well: x' = x
    let t = Term::function(LRA::Id(), [X(x)], [x]).unwrap();
    assert_eq!(t.write().iter().next().map(Wire::id), Some(X(x).id()));
    assert_eq!(t.read().iter().next().map(Wire::id), Some(x.id()));
}

#[test]
fn variable_fail_into_term_constructors() {
    let x = Var::new(Sort::real([1, 1]));
    // dx = x: the derivative wire carries the tangent sort (rank 1), and
    // Id never crosses ranks
    assert!(Term::function(LRA::Id(), [d(x)], [x]).is_err());
}
