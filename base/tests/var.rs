use base::var::{X, X_ref, d, d_ref};
use base::{Term, Var, Wire};
use theory::lra::{LRA, Sort};
use theory::sort::TangentMultiSort;

#[test]
fn variables_are_globally_ordered_by_creation() {
    let x = Var::new(Sort::Real([1, 1]));
    let y = Var::new(Sort::Real([1, 1]));
    let z = Var::new(Sort::Real([1, 1]));

    // the latched wire's id increases with creation
    assert!(x < y && y < z);
    assert!(x <= x && x == x);

    // sorting recovers the creation order
    let mut vars = [z, x, y];
    vars.sort();
    assert_eq!(vars, [x, y, z]);
}

fn borrow_copy_and_compare(x: &Var<Sort>, nxt: Wire<Sort>, der: Wire<Sort>) {
    assert_eq!(d(x), nxt);
    assert_eq!(X(x), der);
}

#[test]
fn variable_casts_to_its_wires() {
    // variables are Copy: owned bindings, ampersand-free use everywhere
    let real = Sort::Real([1, 1]);
    let x = Var::new(real);

    let a: Wire<_> = *x; // the latched wire
    let b: Wire<_> = d(x); //     its derivative
    let c: Wire<_> = X(x); //   the next wire

    // d/nxt copy, never clone and never consume: the variable stays
    // usable and the views are stable
    assert_eq!(d(x), b);
    assert_eq!(X(x), c);

    // three distinct wires; the latched and next carry values, the
    // derivative carries the tangent
    assert_ne!(a, b);
    assert_ne!(a, c);
    assert_ne!(b, c);
    assert_eq!(*a.dtype(), real);
    assert_eq!(*c.dtype(), real);
    assert_eq!(*b.dtype(), real.T());
    assert_eq!(*b.dtype(), Sort::TReal([1, 1], 1));

    // the variable dereferences to its latched wire: `Wire`'s methods
    // apply to it directly, and `&x` coerces to `&Wire<_>`
    assert_eq!(*x.dtype(), real);
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

    let x = Var::new(Sort::Real([1, 1]));

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
    let x = Var::new(Sort::Real([1, 1]));
    // dx = x: the derivative has the tangent sort, not the sort of `x`
    assert!(Term::function(LRA::Id(), [d(x)], [x]).is_err());
}
