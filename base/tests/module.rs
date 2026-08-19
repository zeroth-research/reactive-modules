//! Tests for module-level structure: partial observability and parallel
//! composition.

mod common;

use base::term::Term;
use base::var::{Var, X, d};
use base::wire::Wire;
use common::{Atom, Module, Ops, example_counter, example_peterson1, example_tiny1, mk_op};
use std::fmt;
use theory::{Combinatorial, Differential, Sequential, Theory};

#[test]
fn can_instantiate_partially_observable_module() {
    let (m, _) = example_counter().unwrap();
    let vars = m.obs().clone();
    let mut obs: Vec<Var<&'static str>> = Vec::new();
    let mut prvt: Vec<Var<&'static str>> = Vec::new();
    for var in vars {
        if var.id() == 0 {
            prvt.push(var);
        } else {
            obs.push(var);
        }
    }

    let m = Module::partially_observable(m.atoms().iter().cloned(), &prvt);
    //print!("{}", m);
    assert!(m.is_ok());
}

#[test]
fn cannot_instantiate_external_unobservable_wire() {
    let (m, _) = example_counter().unwrap();
    let vars = m.obs().clone();
    // the fourth observable is an external variable: making it private must
    // fail, privates have to be controlled
    let target = vars.iter().nth(3).unwrap().id();
    let mut obs: Vec<Var<&'static str>> = Vec::new();
    let mut prvt: Vec<Var<&'static str>> = Vec::new();
    for var in vars {
        if var.id() == target {
            prvt.push(var);
        } else {
            obs.push(var);
        }
    }

    let m = Module::partially_observable(m.atoms().iter().cloned(), &prvt);
    print!("{:?}", m);

    assert!(m.is_err());
}

#[test]
fn can_compose_example_peterson1_with_empty_module() {
    let m1 = example_peterson1().unwrap();
    let m2 = Module::empty();

    let _m3 = Module::composition([m1, m2]).unwrap();
}

#[test]
fn can_instantiate_example_tiny1_0123() {
    let x = Var::new("Tny");
    let y = Var::new("Tny");
    let m = example_tiny1(x, y, true).unwrap();
    assert!(m.is_open());
}

#[test]
fn can_instantiate_example_tiny1_2301() {
    let x = Var::new("Tny");
    let y = Var::new("Tny");
    let m = example_tiny1(y, x, true).unwrap();
    assert!(m.is_open());
}

#[test]
fn can_compose_example_tiny1() {
    let x = Var::new("Tny");
    let y = Var::new("Tny");
    let m1 = example_tiny1(x, y, false).unwrap();
    let m2 = example_tiny1(y, x, false).unwrap();

    let m3 = Module::composition([m1, m2]);
    assert!(m3.is_ok());
}

#[test]
fn cannot_compose_example_tiny1_with_cyclic_await() {
    let x = Var::new("Tny");
    let y = Var::new("Tny");
    let m1 = example_tiny1(x, y, true).unwrap();
    let m2 = example_tiny1(y, x, true).unwrap();

    let m3 = Module::composition([m1, m2]);
    assert!(m3.is_err());
}

#[test]
fn can_compose_example_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let x = Var::new("Tny");
    let y = Var::new("Tny");
    let m1 = example_tiny1(x, y, true).unwrap();
    let m2 = example_tiny1(y, x, false).unwrap();

    let m3 = Module::composition([m1, m2]);
    assert!(m3.is_ok());
}

#[test]
fn can_compose_three_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let x = Var::new("Tny");
    let y = Var::new("Tny");
    let z = Var::new("Tny");
    let m1 = example_tiny1(x, y, true).unwrap();
    let m2 = example_tiny1(y, x, false).unwrap();
    let m3 = example_tiny1(y, z, false).unwrap();

    let m4 = Module::composition([m1, m2, m3]);
    assert!(m4.is_ok());
}

#[test]
fn compose_seq() {
    // define two modules:
    //  M1: read external "y" and write it to "x"
    //  M2: read external "x" and write it to "z"
    //
    //  M1 and M2 are compatible (disjoint interface variables and acyclic waiting dependencies),
    //  and we test that they are composable
    let x = Var::new("real");
    let y = Var::new("real");
    let z = Var::new("real");

    let assign: Vec<Term<Ops>> = [term!(mk_op("ID"), [X(x)], [X(y)]).unwrap()].to_vec();
    let m1 = Module::combinatorial(assign, &[x, y]).unwrap();

    let assign: Vec<Term<Ops>> = [term!(mk_op("ID"), [X(z)], [X(x)]).unwrap()].to_vec();
    let m2 = Module::combinatorial(assign, &[x, z]).unwrap();

    Module::composition([m1, m2]).unwrap();
}

/// Composes two coupled modules while hiding their shared variable.
///
/// ```text
///     M1: x' = X(y)   (controls x, reads external y)
///     M2: z' = X(x)   (controls z, reads x)
/// ```
///
/// Hiding `x` at composition keeps the coupling between the components but
/// makes `x` private in the composite: only `z` remains on the interface,
/// with `y` still external. The pure composition `comp` is the special case
/// hiding nothing, and hiding an *external* variable is rejected — privates
/// must be controlled.
#[test]
fn compose_hiding() {
    let x = Var::new("real");
    let y = Var::new("real");
    let z = Var::new("real");

    let assign: Vec<Term<Ops>> = [term!(mk_op("ID"), [X(x)], [X(y)]).unwrap()].to_vec();
    let m1 = Module::combinatorial(assign, &[x, y]).unwrap();

    let assign: Vec<Term<Ops>> = [term!(mk_op("ID"), [X(z)], [X(x)]).unwrap()].to_vec();
    let m2 = Module::combinatorial(assign, &[x, z]).unwrap();

    // hiding the shared variable privatises it in the composite
    let m = Module::hiding_composition([m1.clone(), m2.clone()], &[x]).unwrap();
    let ids = |i: &base::var::Interface<_>| i.iter().map(|v| v.id()).collect::<Vec<_>>();
    assert_eq!(ids(m.prvt()), vec![x.id()]);
    assert_eq!(ids(m.intf()), vec![z.id()]);
    assert_eq!(ids(m.extl()), vec![y.id()]);
    // the hidden variable leaves the observables but stays controlled
    assert_eq!(ids(m.obs()), vec![y.id(), z.id()]);
    assert_eq!(ids(m.ctrl()), vec![x.id(), z.id()]);

    // the pure composition is the special case that hides nothing
    let m = Module::composition([m1.clone(), m2.clone()]).unwrap();
    assert_eq!(m.prvt().len(), 0);
    assert_eq!(ids(m.intf()), vec![x.id(), z.id()]);
    assert_eq!(ids(m.extl()), vec![y.id()]);

    // an external variable cannot be hidden
    assert!(Module::hiding_composition([m1, m2], &[y]).is_err());
}

#[test]
fn compose_seq_2() {
    let x = Var::new("real");
    let y = Var::new("real");
    let z = Var::new("real");
    let y0 = Var::new("real");
    let z0 = Var::new("real");
    let inv = Var::new("real");

    // class Module(smt.Module):
    //     def init(self, extl) -> None:
    //         y0, z0 = extl
    //         return Int(0), X(y0), X(z0)  # = x, y, z
    //
    //     def update(self, ctrl, extl) -> None:
    //         x, y, z = ctrl
    //
    //         cond = Or(x < y, x < z)
    //         xn = Ite(cond, x + Int(1), Int(0))
    //
    //         return xn, y, z
    //
    let init: Vec<Term<Ops>> = [
        term!(mk_op("Const(0)"), [X(x)]).unwrap(),
        term!(mk_op("Id"), [X(y)], [X(y0)]).unwrap(),
        term!(mk_op("Id"), [X(z)], [X(z0)]).unwrap(),
    ]
    .to_vec();

    let tmps = [
        Wire::zero("real"),
        Wire::zero("real"),
        Wire::zero("real"),
        Wire::zero("real"),
        Wire::zero("real"),
        Wire::zero("real"),
    ];
    let update: Vec<Term<Ops>> = [
        term!(mk_op("Lt"), [tmps[0]], [x, y]).unwrap(),
        term!(mk_op("Lt"), [tmps[1]], [x, z]).unwrap(),
        term!(mk_op("Or"), [tmps[2]], [tmps[0], tmps[1]]).unwrap(),
        term!(mk_op("Const(0)"), [tmps[3]]).unwrap(),
        term!(mk_op("Const(1)"), [tmps[4]]).unwrap(),
        term!(mk_op("Add"), [tmps[5]], [*x, tmps[4]]).unwrap(),
        term!(mk_op("Ite"), [X(x)], [tmps[2], tmps[5], tmps[3]]).unwrap(),
        term!(mk_op("Id"), [X(y)], [y]).unwrap(),
        term!(mk_op("Id"), [X(z)], [z]).unwrap(),
    ]
    .to_vec();
    let obs = &[x, y, z, y0, z0];
    let m1 = Module::sequential(init, update, obs, []).unwrap();

    //
    // class Inv(smt.Module):
    //     def init(self, extl) -> None:
    //         x, y, z = extl
    //         return Or(X(x) <= X(y), X(x) <= X(z))
    //
    //     def update(self, inv, extl) -> None:
    //         x, y, z = extl
    //         return Or(X(x) <= X(y), X(x) <= X(z))
    let tmps = [Wire::zero("real"), Wire::zero("real"), Wire::zero("real")];
    let assign: Vec<Term<Ops>> = [
        term!(mk_op("Le"), [tmps[0]], [X(x), X(y)]).unwrap(),
        term!(mk_op("Le"), [tmps[1]], [X(x), X(z)]).unwrap(),
        term!(mk_op("Or"), [tmps[2]], [tmps[0], tmps[1]]).unwrap(),
        term!(mk_op("Id"), [X(inv)], [tmps[2]]).unwrap(),
    ]
    .to_vec();

    let obs = &[x, y, z, inv];
    let m2 = Module::combinatorial(assign.clone(), obs).unwrap();

    Module::composition([m1.clone(), m2]).unwrap();

    // try to use a `sequential_observable` ctor instead of combinatorial
    let m2 = Module::sequential(assign.clone(), assign, obs, []).unwrap();
    let _m = Module::composition([m1, m2]).unwrap();
    println!("{:?}", _m);
}

#[test]
fn more_controlled_than_external() {
    let x = Var::new("A");
    let y = Var::new("B");
    let z = Var::new("C");

    let init = Term::constant(mk_op("A"), [X(y), X(z)]).unwrap();
    let update = Term::function(mk_op("A"), [X(y), X(z)], [y, z]).unwrap();

    let a = Atom::sequential(&[x, y, z], [init], [update]);
    assert!(a.is_ok());
    let m = Module::partially_observable([a.unwrap()], &[y, z]);
    assert!(m.is_ok());
}

#[allow(unused)]
#[derive(Clone, Copy)]
struct SeqOps(&'static str);
#[allow(unused)]
#[derive(Clone, Copy)]
struct DifOps(&'static str);

impl Theory for SeqOps {
    type Sort = &'static str;
    const NAME: &'static str = "SeqOps";
    fn check<R, W, E: fmt::Display>(&self, _read: R, _write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<(Self::Sort, u8), E>>,
        W: IntoIterator<Item = Result<(Self::Sort, u8), E>>,
    {
        Ok(())
    }
}

impl Theory for DifOps {
    type Sort = &'static str;
    const NAME: &'static str = "DifOps";
    fn check<R, W, E: fmt::Display>(&self, _read: R, _write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<(Self::Sort, u8), E>>,
        W: IntoIterator<Item = Result<(Self::Sort, u8), E>>,
    {
        Ok(())
    }
}

impl Differential for DifOps {
    const ZERO: Self = Self("ZERO");
}
impl Sequential for SeqOps {
    const SKIP: Self = Self("SKIP");
}

impl Combinatorial for SeqOps {
    const HAVOC: Self = Self("HAVOC");
}

impl fmt::Display for DifOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl fmt::Display for SeqOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[test]
#[allow(non_snake_case)]
fn heterogeneous_composition() {
    let x = Var::new("A");
    let y = Var::new("B");
    let z = Var::new("C");

    let init = Term::constant(SeqOps::HAVOC, [X(x)]).unwrap();
    let jump = Term::function(SeqOps::SKIP, [X(x)], [x]).unwrap();
    let P = base::Module::sequential([init], [jump], &[x], []).unwrap();

    let init = Term::constant(SeqOps::HAVOC, [X(y)]).unwrap();
    let flow = Term::constant(DifOps::ZERO, [d(y)]).unwrap();
    let Q = base::Module::differential([init], [flow], &[y], []).unwrap();

    let comb = Term::function(SeqOps("+"), [X(z)], [X(x), X(y)]).unwrap();
    let R = base::Module::combinatorial([comb], &[x, y, z]).unwrap();

    let S = base::Module::composition([P, Q, R]);
    assert!(S.is_ok());
}
