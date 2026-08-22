//! Tests for atomic modules: modules made of a single atom, built through
//! the `sequential`, `differential`, and related constructors.

mod common;

use base::term::Term;
use base::var::{Var, X, d};
use base::wire::Wire;
use common::{Atom, Module, Ops, example_counter, example_peterson1, mk_op};

#[test]
fn can_instantiate_and_print_sequential_module() {
    let (module, names) = example_counter().unwrap();
    // the naming function makes totality explicit: unnamed variables fall
    // back to their id instead of panicking
    print!("{}", module.with_varnames(|v| names[v].as_str()));

    let atom = &module.atoms()[0];
    print!("{}", atom.with_varnames(|v| names[v].as_str()));
}

#[test]
fn can_instantiate_example_peterson1() {
    let m = example_peterson1().unwrap();

    assert!(m.is_open());
    assert_eq!(m.extl().len(), 2);
    assert_eq!(m.intf().len(), 2);
}

#[test]
fn module_write_all_ctrl() {
    let x = Var::new("real");
    let y = Var::new("real");
    let x0 = Var::new("real");

    let update: Vec<Term<Ops>> = [term!(mk_op("ID"), [X(x)], [x]).unwrap()].to_vec();

    let obs = &[x, y, x0];

    let m = Module::sequential(obs, vec![], update.clone());
    assert!(m.is_err());

    let init: Vec<Term<Ops>> = [term!(mk_op("ID"), [X(x0)], [X(x)]).unwrap()].to_vec();
    let m = Module::sequential(obs, init, update.clone());
    assert!(m.is_err());

    let init: Vec<Term<Ops>> = [
        term!(mk_op("ID"), [X(x)], [X(x0)]).unwrap(),
        term!(mk_op("ID"), [X(y)], [X(x0)]).unwrap(),
    ]
    .to_vec();

    let m = Module::sequential(obs, init.clone(), update);
    assert!(m.is_err());

    let update: Vec<Term<Ops>> = [
        term!(mk_op("ID"), [X(x)], [x]).unwrap(),
        term!(mk_op("ID"), [X(y)], [y]).unwrap(),
    ]
    .to_vec();

    let m = Module::sequential(obs, init, update);
    assert!(m.is_ok());
}

#[test]
fn atom_with_invalid_read() {
    let x = Var::new("A");
    let y = Var::new("B");
    let z = Var::new("C");
    let p = Wire::zero("P");

    let init = Term::function(mk_op("i"), [X(y), X(z)], [p, X(x)]).unwrap();
    let update = Term::function(mk_op("u"), [X(y), X(z)], [p, X(x), *y, *z]).unwrap();

    let a = Atom::sequential(&[x, y, z], [init], [update]);
    println!("{:?}", a);
    assert!(a.is_err());
    // the failure carries over: an invalid atom cannot be promoted to a module
    let m = a.and_then(|a| Module::partially_observable([a], |v| v == &y || v == &z));
    assert!(m.is_err());
}

/// Builds an open differential module encoding the drifting clocks
///
/// ```text
///     dx = 2 dt
///     dy = 1 dt
/// ```
///
/// where `t` is an *input*: the module does not control it, so it is
/// inferred external and the module is open. The clocks `x` and `y` drift at
/// rates 2 and 1 relative to `t`, expressed by delay terms that *await* the
/// input's derivative `d(t)`: `d(x) = d(t) + d(t)` and `d(y) = Id(d(t))`.
/// Since the rates are relative to the input clock rather than constant in
/// absolute time, these are drifting clocks and not a linear ODE.
#[test]
fn differential_drifting_clocks() {
    use std::collections::HashSet;
    use theory::lra::{LRA, Sort};

    // Real scalars: 1x1 matrices.
    let scalar = Sort::Real([1, 1]);
    // The input clock, and the two clocks drifting relative to it.
    let t = Var::new(scalar);
    let x = Var::new(scalar);
    let y = Var::new(scalar);

    // delay: the derivatives, both driven by the input's derivative `d(t)`.
    let delay = [
        // dx = 2 dt, synthesised tensor-free as dt + dt
        Term::function(LRA::Add(), [d(x)], [d(t), d(t)]).unwrap(),
        // dy = 1 dt
        Term::function(LRA::Id(), [d(y)], [d(t)]).unwrap(),
    ];

    // init: the initial derivatives are left unconstrained (Havoc). Every
    // controlled wire must be written by the init block too.
    let init = [
        Term::constant(LRA::AnyReal([1, 1]), [X(x)]).unwrap(),
        Term::constant(LRA::AnyReal([1, 1]), [X(y)]).unwrap(),
    ];

    // The atom writes only d(x) and d(y), so `t` is inferred external.
    let obs = &[x, y, t];

    let module: base::Module<LRA, LRA, LRA> = base::Module::differential(obs, init, delay)
        .expect("differential module should be well-formed");

    // An open module: the input clock is external.
    assert!(module.is_open());
    let extl: HashSet<usize> = module.extl().iter().map(|v| v.id()).collect();
    assert_eq!(extl, HashSet::from([t.id()]));

    // The clocks are controlled by the module, hence interface variables.
    let intf: HashSet<usize> = module.intf().iter().map(|v| v.id()).collect();
    assert_eq!(intf, HashSet::from([x.id(), y.id()]));

    // The differential constructor produces exactly one atom.
    assert_eq!(module.atoms().len(), 1);
    let atom = &module.atoms()[0];

    // The atom controls the clocks and awaits the input.
    let ctrl: HashSet<usize> = atom.ctrl().iter().map(|v| v.id()).collect();
    assert_eq!(ctrl, HashSet::from([x.id(), y.id()]));

    let wait: HashSet<usize> = atom.wait().iter().map(|v| v.id()).collect();
    assert_eq!(wait, HashSet::from([t.id()]));

    // Block sizes: 2 derivative terms, 2 init terms, and the update is the
    // implicit `skip` synthesised by the constructor, one term per
    // controlled variable.
    assert_eq!(atom.delay().len(), 2);
    assert_eq!(atom.init().len(), 2);
    assert_eq!(atom.update().len(), 2);

    // All three variables are observable.
    assert_eq!(module.obs().len(), 3);
}

/// Proposes a differential module encoding
///
/// ```text
///     dx = x
/// ```
///
/// and expects it to be rejected: the delay term drives the derivative
/// `d(x)` (degree 1) directly from the value wire `x` (degree 0), and
/// ordinary operations require uniform degrees across their operands.
#[test]
fn differential_module_rejects_dx_equals_x() {
    use theory::lra::{LRA, Sort};

    let scalar = Sort::Real([1, 1]);
    let x = Var::new(scalar);

    // dx = x: already ill-formed at the term level.
    let delay = Term::function(LRA::Id(), [d(x)], [x]);

    let module = delay.and_then(|delay| {
        let init = [Term::constant(LRA::AnyReal([1, 1]), [X(x)]).unwrap()];
        base::Module::<LRA, LRA, LRA>::differential(&[x], init, [delay])
    });
    assert!(module.is_err());
}

/// A jump module: only the update is given, the initial state is
/// havoced and the delay is zero, both synthesised by the constructor.
#[test]
fn uninitialized_module_ok() {
    let x = Var::new("real");
    let y = Var::new("real");

    // x' = y: x is controlled, y is inferred external.
    let update = [Term::function(mk_op("ID"), [X(x)], [y]).unwrap()];
    let m = Module::jump(&[x, y], update).unwrap();

    assert!(m.is_open());
    assert_eq!(m.atoms().len(), 1);
    let atom = &m.atoms()[0];

    // One synthesised HAVOC init and one synthesised ZERO delay for the
    // single controlled variable, next to the explicit update.
    assert_eq!(atom.init().len(), 1);
    assert_eq!(atom.update().len(), 1);
    assert_eq!(atom.delay().len(), 1);
}

/// The update of an jump module must drive next wires; writing a
/// latched wire is rejected.
#[test]
fn uninitialized_module_rejects_latched_write() {
    let x = Var::new("real");
    let y = Var::new("real");

    // writes the latched wire `x` instead of the next wire `X(x)`
    let update = [Term::function(mk_op("ID"), [*x], [y]).unwrap()];
    assert!(Module::jump(&[x, y], update).is_err());
}

/// A constant module: only the init is given, the variable is held by a
/// synthesised SKIP update and a synthesised ZERO delay.
#[test]
fn constant_module_ok() {
    let x = Var::new("real");

    let init = [Term::constant(mk_op("CONST(0)"), [X(x)]).unwrap()];
    let m = Module::constant(&[x], init).unwrap();

    assert!(m.is_closed());
    assert_eq!(m.atoms().len(), 1);
    let atom = &m.atoms()[0];

    // The explicit init, next to one synthesised SKIP update and one
    // synthesised ZERO delay for the single controlled variable.
    assert_eq!(atom.init().len(), 1);
    assert_eq!(atom.update().len(), 1);
    assert_eq!(atom.delay().len(), 1);
}

/// The init of a constant module must drive next wires; writing a latched
/// wire is rejected.
#[test]
fn constant_module_rejects_latched_write() {
    let x = Var::new("real");

    // writes the latched wire `x` instead of the next wire `X(x)`
    let init = [Term::constant(mk_op("CONST(0)"), [*x]).unwrap()];
    assert!(Module::constant(&[x], init).is_err());
}

/// A hybrid module encoding a simple timed automaton: a switch with one
/// clock.
///
/// ```text
///     loc ∈ {off, on},  clock x
///
///     init:  loc = off, x = 0
///     flow:  dx = 1, dloc = 0
///     jump:  when x >= 1, toggle loc and reset x; otherwise hold
/// ```
///
/// All three blocks are explicit: the discrete jump drives the next wires,
/// the continuous flow drives the derivatives — the clock at rate 1, the
/// location with zero drift, since it only changes at jumps.
#[test]
fn simple_timed_automaton() {
    let x = Var::new("real");
    let loc = Var::new("{off, on}");

    // init: loc = off, x = 0
    let init = [
        Term::constant(mk_op("CONST(off)"), [X(loc)]).unwrap(),
        Term::constant(mk_op("CONST(0)"), [X(x)]).unwrap(),
    ];

    // jump: guard x >= 1; toggle the location and reset the clock when it
    // fires, hold both otherwise
    let guard = Wire::zero("bool");
    let update = [
        Term::function(mk_op("GEQ(1)"), [guard], [x]).unwrap(),
        Term::function(mk_op("TOGGLE_IF"), [X(loc)], [guard, *loc]).unwrap(),
        Term::function(mk_op("RESET_IF"), [X(x)], [guard, *x]).unwrap(),
    ];

    // flow: dx = 1, dloc = 0
    let delay = [
        Term::constant(mk_op("ONE"), [d(x)]).unwrap(),
        Term::constant(mk_op("ZERO"), [d(loc)]).unwrap(),
    ];

    let m = Module::hybrid(&[x, loc], init, update, delay).unwrap();

    // A closed module controlling both the clock and the location.
    assert!(m.is_closed());
    assert_eq!(m.intf().len(), 2);
    assert_eq!(m.atoms().len(), 1);
    let atom = &m.atoms()[0];
    assert_eq!(atom.ctrl().len(), 2);

    // Nothing is synthesised: exactly the three explicit blocks, with the
    // guard as a temporary wire of the update.
    assert_eq!(atom.init().len(), 2);
    assert_eq!(atom.update().len(), 3);
    assert_eq!(atom.delay().len(), 2);
}

/// A hybrid module synthesises nothing: forgetting the zero drift of the
/// discrete location leaves its derivative unwritten, and the timed
/// automaton is rejected.
#[test]
fn hybrid_rejects_missing_flow() {
    let x = Var::new("real");
    let loc = Var::new("{off, on}");

    let init = [
        Term::constant(mk_op("CONST(off)"), [X(loc)]).unwrap(),
        Term::constant(mk_op("CONST(0)"), [X(x)]).unwrap(),
    ];

    let guard = Wire::zero("bool");
    let update = [
        Term::function(mk_op("GEQ(1)"), [guard], [x]).unwrap(),
        Term::function(mk_op("TOGGLE_IF"), [X(loc)], [guard, *loc]).unwrap(),
        Term::function(mk_op("RESET_IF"), [X(x)], [guard, *x]).unwrap(),
    ];

    // flow: dx = 1, but no dloc = 0
    let delay = [Term::constant(mk_op("ONE"), [d(x)]).unwrap()];

    let m = Module::hybrid(&[x, loc], init, update, delay);
    assert!(m.is_err());
}
