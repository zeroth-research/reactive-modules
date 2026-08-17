use base::term;
use base::term::Term;
use base::variable::{Variable, X, d};
use base::wire::Wire;
use std::fmt;
use theory::{Combinatorial, Differential, Sequential, Theory};

#[derive(Clone, Debug)]
#[allow(unused)]
struct Ops(&'static str);

impl Theory for Ops {
    type Sort = &'static str;
    const NAME: &'static str = "Ops";

    fn check<R, W, E: fmt::Display>(&self, _read: R, _write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = Result<(Self::Sort, u8), E>>,
        W: IntoIterator<Item = Result<(Self::Sort, u8), E>>,
    {
        Ok(())
    }
}

impl Combinatorial for Ops {
    const HAVOC: Self = Ops("HAVOC");
}

impl Sequential for Ops {
    const SKIP: Self = Ops("SKIP");
}

impl Differential for Ops {
    const ZERO: Self = Ops("ZERO");
}

type Module = base::Module<Ops, Ops, Ops>;
type Atom = base::Atom<Ops, Ops, Ops>;

fn mk_op(name: &'static str) -> Ops {
    Ops(name)
}

#[allow(clippy::vec_init_then_push)]
fn example_counter() -> Result<Module, String> {
    let x = Variable::new("real");
    let y = Variable::new("real");
    let z = Variable::new("real");
    let y0 = Variable::new("real");
    let z0 = Variable::new("real");

    let mut init: Vec<Term<Ops>> = Vec::new();

    let tmp1 = Wire::zero("real");
    init.push(term!(mk_op("ZERO"), [tmp1])?);

    let tmp2 = Wire::zero("bool");
    let tmp3 = Wire::zero("bool");
    init.push(term!(mk_op("ID"), [X(x)], [tmp1])?);
    init.push(term!(mk_op("ABS"), [tmp2], [X(y0)])?);
    init.push(term!(mk_op("ID"), [X(y)], [tmp2])?);
    init.push(term!(mk_op("ABS"), [tmp3], [X(z0)])?);
    init.push(term!(mk_op("ID"), [X(z)], [tmp3])?);

    let mut update: Vec<Term<Ops>> = Vec::new();

    let tmp4 = Wire::zero("bool");
    let tmp5 = Wire::zero("bool");
    let tmp6 = Wire::zero("bool");
    update.push(term!(mk_op("ZERO"), [tmp1])?);
    update.push(term!(mk_op("LEQ"), [tmp4], [x, y])?);
    update.push(term!(mk_op("LEQ"), [tmp5], [x, z])?);
    update.push(term!(mk_op("OR"), [tmp6], [tmp4, tmp5])?);

    let tmp7 = Wire::zero("real");
    let tmp8 = Wire::zero("real");
    update.push(term!(mk_op("ONE"), [tmp7])?);
    update.push(term!(mk_op("ADD"), [tmp8], [*x, tmp7])?);

    update.push(term!(mk_op("ITE"), [X(x)], [tmp6, tmp8, tmp1])?);
    update.push(term!(mk_op("ID"), [X(y)], [y0])?);
    update.push(term!(mk_op("ID"), [X(z)], [z0])?);

    let obs = [x, y, z, y0, z0];

    Module::sequential(obs, init, update)
}

#[allow(clippy::vec_init_then_push)]
fn example_peterson1() -> Result<Module, String> {
    let stype = "{outCS, reqCS, inCS}";
    let pc1 = Variable::new(stype);
    let x1 = Variable::new("bool");
    let pc2 = Variable::new(stype);
    let x2 = Variable::new("bool");

    let mut init: Vec<Term<Ops>> = Vec::new();
    init.push(term!(mk_op("CONST(outCS)"), [X(pc1)]).unwrap());
    init.push(term!(mk_op("CONST(true)"), [X(x1)]).unwrap());

    let mut update: Vec<Term<Ops>> = Vec::new();
    let out_cs = Wire::zero(stype);
    let cond1 = Wire::zero("bool");
    update.push(term!(mk_op("CONST(outCS)"), [out_cs]).unwrap());
    update.push(term!(mk_op("EQ"), [cond1], [out_cs, *pc1]).unwrap());

    let req_cs = Wire::zero(stype);
    let cond2 = Wire::zero("bool");
    update.push(term!(mk_op("CONST(reqCS)"), [req_cs]).unwrap());
    let tmp11 = Wire::zero("bool");
    update.push(term!(mk_op("EQ"), [tmp11], [req_cs, *pc1]).unwrap());

    let tmp12 = Wire::zero("bool");
    let tmp13 = Wire::zero("bool");
    let tmp14 = Wire::zero("bool");
    update.push(term!(mk_op("EQ"), [tmp12], [out_cs, *pc2]).unwrap());
    update.push(term!(mk_op("NEQ"), [tmp13], [x1, x2]).unwrap());
    update.push(term!(mk_op("OR"), [tmp14], [tmp12, tmp13]).unwrap());
    update.push(term!(mk_op("AND"), [cond2], [tmp14, tmp11]).unwrap());

    let in_cs = Wire::zero(stype);
    let cond3 = Wire::zero("bool");
    update.push(term!(mk_op("CONST(inCS)"), [in_cs]).unwrap());
    update.push(term!(mk_op("EQ"), [cond3], [in_cs, *pc1]).unwrap());

    let const_true = Wire::zero("bool");
    update.push(term!(mk_op("CONST(true)"), [const_true]).unwrap());

    update.push(
        term!(
            mk_op("CASE"),
            [X(pc1), X(x1)],
            [
                cond1, req_cs, *x2, cond2, in_cs, *x1, cond3, out_cs, *x1, const_true, *pc1, *x1,
            ]
        )
        .unwrap(),
    );

    let obs = [pc1, x1, pc2, x2];
    Module::sequential(obs, init, update)
}

fn example_tiny1(
    external: Variable<&'static str>,
    interface: Variable<&'static str>,
    wait: bool,
) -> Result<Module, String> {
    let private = Variable::new("Tny");
    let temp = Wire::zero("Tny");

    let cons = Term::constant(mk_op("CONST"), [temp]).unwrap();

    let update = if wait {
        Term::function(
            mk_op("AWAIT"),
            [X(interface), X(private)],
            [X(external), *private, temp],
        )
        .unwrap()
    } else {
        Term::function(
            mk_op("SEQ"),
            [X(interface), X(private)],
            [*external, *private, temp],
        )
        .unwrap()
    };

    let init = Term::constant(mk_op("INIT"), [X(interface), X(private)]).unwrap();

    let vars = [external, interface, private];
    let obs = [external, interface];
    let prvt = [private];

    let atom = Atom::sequential(&vars, [init], [cons, update])?;
    Module::partially_observable(obs, prvt, [atom])
}
//
#[test]
fn can_instantiate_sequential_module() {
    let _module = example_counter().unwrap();
    //print!("{}", _module);
}

#[test]
fn can_instantiate_partially_observable_module() {
    let m = example_counter().unwrap();
    let vars = m.obs().clone();
    let mut obs: Vec<Variable<&'static str>> = Vec::new();
    let mut prvt: Vec<Variable<&'static str>> = Vec::new();
    for var in vars {
        if var.id() == 0 {
            prvt.push(var);
        } else {
            obs.push(var);
        }
    }

    let m = Module::partially_observable(obs, prvt, m.atoms().iter().cloned());
    //print!("{}", m);
    assert!(m.is_ok());
}

#[test]
fn cannot_instantiate_external_unobservable_wire() {
    let m = example_counter().unwrap();
    let vars = m.obs().clone();
    // the fourth observable is an external variable: making it private must
    // fail, privates have to be controlled
    let target = vars.iter().nth(3).unwrap().id();
    let mut obs: Vec<Variable<&'static str>> = Vec::new();
    let mut prvt: Vec<Variable<&'static str>> = Vec::new();
    for var in vars {
        if var.id() == target {
            prvt.push(var);
        } else {
            obs.push(var);
        }
    }

    let m = Module::partially_observable(obs, prvt, m.atoms().iter().cloned());
    print!("{:?}", m);

    assert!(m.is_err());
}

#[test]
fn can_instantiate_example_peterson1() {
    let m = example_peterson1().unwrap();

    assert!(m.is_open());
    assert_eq!(m.extl().len(), 2);
    assert_eq!(m.intf().len(), 2);
}
//
#[test]
fn module_write_all_ctrl() {
    let x = Variable::new("real");
    let y = Variable::new("real");
    let x0 = Variable::new("real");

    let update: Vec<Term<Ops>> = [term!(mk_op("ID"), [X(x)], [x]).unwrap()].to_vec();

    let obs = [x, y, x0];

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
fn can_compose_example_peterson1_with_empty_module() {
    let m1 = example_peterson1().unwrap();
    let m2 = Module::empty();

    let _m3 = Module::parallel([m1, m2]).unwrap();
}

#[test]
fn can_instantiate_example_tiny1_0123() {
    let x = Variable::new("Tny");
    let y = Variable::new("Tny");
    let m = example_tiny1(x, y, true).unwrap();
    assert!(m.is_open());
}

#[test]
fn can_instantiate_example_tiny1_2301() {
    let x = Variable::new("Tny");
    let y = Variable::new("Tny");
    let m = example_tiny1(y, x, true).unwrap();
    assert!(m.is_open());
}
#[test]
fn can_compose_example_tiny1() {
    let x = Variable::new("Tny");
    let y = Variable::new("Tny");
    let m1 = example_tiny1(x, y, false).unwrap();
    let m2 = example_tiny1(y, x, false).unwrap();

    let m3 = Module::parallel([m1, m2]);
    assert!(m3.is_ok());
}

#[test]
fn cannot_compose_example_tiny1_with_cyclic_await() {
    let x = Variable::new("Tny");
    let y = Variable::new("Tny");
    let m1 = example_tiny1(x, y, true).unwrap();
    let m2 = example_tiny1(y, x, true).unwrap();

    let m3 = Module::parallel([m1, m2]);
    assert!(m3.is_err());
}

#[test]
fn can_compose_example_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let x = Variable::new("Tny");
    let y = Variable::new("Tny");
    let m1 = example_tiny1(x, y, true).unwrap();
    let m2 = example_tiny1(y, x, false).unwrap();

    let m3 = Module::parallel([m1, m2]);
    assert!(m3.is_ok());
}

#[test]
fn can_compose_three_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let x = Variable::new("Tny");
    let y = Variable::new("Tny");
    let z = Variable::new("Tny");
    let m1 = example_tiny1(x, y, true).unwrap();
    let m2 = example_tiny1(y, x, false).unwrap();
    let m3 = example_tiny1(y, z, false).unwrap();

    let m4 = Module::parallel([m1, m2, m3]);
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
    let x = Variable::new("real");
    let y = Variable::new("real");
    let z = Variable::new("real");

    let assign: Vec<Term<Ops>> = [term!(mk_op("ID"), [X(x)], [X(y)]).unwrap()].to_vec();
    let m1 = Module::combinatorial([x, y], assign).unwrap();

    let assign: Vec<Term<Ops>> = [term!(mk_op("ID"), [X(z)], [X(x)]).unwrap()].to_vec();
    let m2 = Module::combinatorial([x, z], assign).unwrap();

    Module::parallel([m1, m2]).unwrap();
}

#[test]
fn compose_seq_2() {
    let x = Variable::new("real");
    let y = Variable::new("real");
    let z = Variable::new("real");
    let y0 = Variable::new("real");
    let z0 = Variable::new("real");
    let inv = Variable::new("real");

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
    let obs = [x, y, z, y0, z0];
    let m1 = Module::sequential(obs, init, update).unwrap();

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

    let obs = [x, y, z, inv];
    let m2 = Module::combinatorial(obs, assign.clone()).unwrap();

    Module::parallel([m1.clone(), m2]).unwrap();

    // try to use a `sequential_observable` ctor instead of combinatorial
    let m2 = Module::sequential(obs, assign.clone(), assign).unwrap();
    let _m = Module::parallel([m1, m2]).unwrap();
    println!("{:?}", _m);
}

#[test]
fn more_controlled_than_external() {
    let x = Variable::new("A");
    let y = Variable::new("B");
    let z = Variable::new("C");

    let init = Term::constant(mk_op("A"), [X(y), X(z)]).unwrap();
    let update = Term::function(mk_op("A"), [X(y), X(z)], [y, z]).unwrap();

    let a = Atom::sequential(&[x, y, z], [init], [update]);
    assert!(a.is_ok());
    let m = Module::partially_observable([x], [y, z], [a.unwrap()]);
    assert!(m.is_ok());
}

#[test]
fn atom_with_invalid_read() {
    let x = Variable::new("A");
    let y = Variable::new("B");
    let z = Variable::new("C");
    let p = Wire::zero("P");

    let init = Term::function(mk_op("i"), [X(y), X(z)], [p, X(x)]).unwrap();
    let update = Term::function(mk_op("u"), [X(y), X(z)], [p, X(x), *y, *z]).unwrap();

    let a = Atom::sequential(&[x, y, z], [init], [update]);
    println!("{:?}", a);
    assert!(a.is_err());
    // let m = Module::partially_observable([x], [y, z], [a.unwrap()]);
    // assert!(m.is_err());
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
    let x = Variable::new("A");
    let y = Variable::new("B");
    let z = Variable::new("C");

    let init = Term::constant(SeqOps::HAVOC, [X(x)]).unwrap();
    let jump = Term::function(SeqOps::SKIP, [X(x)], [x]).unwrap();
    let P = base::Module::sequential([x], [init], [jump]).unwrap();

    let init = Term::constant(SeqOps::HAVOC, [X(y)]).unwrap();
    let flow = Term::constant(DifOps::ZERO, [d(y)]).unwrap();
    let Q = base::Module::differential([y], [init], [flow]).unwrap();

    let comb = Term::function(SeqOps("+"), [X(z)], [X(x), X(y)]).unwrap();
    let R = base::Module::combinatorial([x, y, z], [comb]).unwrap();

    let S = base::Module::parallel([P, Q, R]);
    assert!(S.is_ok());

    print!("{}", S.unwrap());
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
    let t = Variable::new(scalar);
    let x = Variable::new(scalar);
    let y = Variable::new(scalar);

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
        Term::constant(LRA::Havoc(), [X(x)]).unwrap(),
        Term::constant(LRA::Havoc(), [X(y)]).unwrap(),
    ];

    // The atom writes only d(x) and d(y), so `t` is inferred external.
    let obs = [x, y, t];

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
    let x = Variable::new(scalar);

    // dx = x: already ill-formed at the term level.
    let delay = Term::function(LRA::Id(), [d(x)], [x]);

    let module = delay.and_then(|delay| {
        let init = [Term::constant(LRA::Havoc(), [X(x)]).unwrap()];
        base::Module::<LRA, LRA, LRA>::differential([x], init, [delay])
    });
    assert!(module.is_err());
}
