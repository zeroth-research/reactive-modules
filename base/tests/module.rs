use base::term;
use base::term::Term;
use base::wire::Interface;
use base::wire::Wire;
use std::fmt;
use theory::{Combinatorial, Differential, Sequential, Theory};

#[derive(Clone, Debug)]
#[allow(unused)]
struct Ops(&'static str);

impl Theory for Ops {
    type Sort = &'static str;
    const NAME: &'static str = "Ops";

    fn check<R, W, S, E: fmt::Display>(&self, _read: R, _write: W) -> Result<(), String>
    where
        S: TryInto<Self::Sort, Error = E>,
        R: IntoIterator<Item = S>,
        W: IntoIterator<Item = S>,
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

fn mk_op(name: &'static str) -> Ops {
    Ops(name)
}

#[allow(clippy::vec_init_then_push)]
fn example_counter() -> Result<Module, String> {
    let x0 = Wire::new("real");
    let y0 = Wire::new("real");
    let z0 = Wire::new("real");
    let y00 = Wire::new("real");
    let z00 = Wire::new("real");
    let x1 = Wire::new("real");
    let y1 = Wire::new("real");
    let z1 = Wire::new("real");
    let y01 = Wire::new("real");
    let z01 = Wire::new("real");

    let mut init: Vec<Term<Ops>> = Vec::new();

    let tmp1 = Wire::new("real");
    init.push(term!(mk_op("ZERO"), [tmp1.clone()])?);

    let tmp2 = Wire::new("bool");
    let tmp3 = Wire::new("bool");
    init.push(term!(mk_op("ID"), [x1.clone()], [tmp1.clone()])?);
    init.push(term!(mk_op("ABS"), [tmp2.clone()], [y01.clone()])?);
    init.push(term!(mk_op("ID"), [y1.clone()], [tmp2.clone()])?);
    init.push(term!(mk_op("ABS"), [tmp3.clone()], [z01.clone()])?);
    init.push(term!(mk_op("ID"), [z1.clone()], [tmp3.clone()])?);

    let mut update: Vec<Term<Ops>> = Vec::new();

    let tmp4 = Wire::new("bool");
    let tmp5 = Wire::new("bool");
    let tmp6 = Wire::new("bool");
    update.push(term!(mk_op("ZERO"), [tmp1.clone()])?);
    update.push(term!(
        mk_op("LEQ"),
        [tmp4.clone()],
        [x0.clone(), y0.clone()]
    )?);
    update.push(term!(
        mk_op("LEQ"),
        [tmp5.clone()],
        [x0.clone(), z0.clone()]
    )?);
    update.push(term!(
        mk_op("OR"),
        [tmp6.clone()],
        [tmp4.clone(), tmp5.clone()]
    )?);

    let tmp7 = Wire::new("real");
    let tmp8 = Wire::new("real");
    update.push(term!(mk_op("ONE"), [tmp7.clone()])?);
    update.push(term!(
        mk_op("ADD"),
        [tmp8.clone()],
        [x0.clone(), tmp7.clone()]
    )?);

    update.push(term!(
        mk_op("ITE"),
        [x1.clone()],
        [tmp6.clone(), tmp8.clone(), tmp1.clone()]
    )?);
    update.push(term!(mk_op("ID"), [y1.clone()], [y0.clone()])?);
    update.push(term!(mk_op("ID"), [z1.clone()], [z0.clone()])?);

    let obs = Interface::from_iter([[x0, x1], [y0, y1], [z0, z1], [y00, y01], [z00, z01]]);

    Module::sequential(obs, init, update)
}

#[allow(clippy::vec_init_then_push)]
fn example_peterson1() -> Result<Module, String> {
    let stype = "{outCS, reqCS, inCS}";
    let pc1: [Wire<&str>; 2] = [Wire::new(stype), Wire::new(stype)];
    let x1: [Wire<&str>; 2] = [Wire::new("bool"), Wire::new("bool")].map(Into::into);
    let pc2: [Wire<&str>; 2] = [Wire::new(stype), Wire::new(stype)].map(Into::into);
    let x2: [Wire<&str>; 2] = [Wire::new("bool"), Wire::new("bool")].map(Into::into);

    let mut init: Vec<Term<Ops>> = Vec::new();
    init.push(term!(mk_op("CONST(outCS)"), [pc1[1].clone()]).unwrap());
    init.push(term!(mk_op("CONST(true)"), [x1[1].clone()]).unwrap());

    let mut update: Vec<Term<Ops>> = Vec::new();
    let out_cs = Wire::new(stype);
    let cond1 = Wire::new("bool");
    update.push(term!(mk_op("CONST(outCS)"), [out_cs.clone()]).unwrap());
    update.push(
        term!(
            mk_op("EQ"),
            [cond1.clone()],
            [out_cs.clone(), pc1[0].clone()]
        )
        .unwrap(),
    );

    let req_cs = Wire::new(stype);
    let cond2 = Wire::new("bool");
    update.push(term!(mk_op("CONST(reqCS)"), [req_cs.clone()]).unwrap());
    let tmp11 = Wire::new("bool");
    update.push(
        term!(
            mk_op("EQ"),
            [tmp11.clone()],
            [req_cs.clone(), pc1[0].clone()]
        )
        .unwrap(),
    );

    let tmp12 = Wire::new("bool");
    let tmp13 = Wire::new("bool");
    let tmp14 = Wire::new("bool");
    update.push(
        term!(
            mk_op("EQ"),
            [tmp12.clone()],
            [out_cs.clone(), pc2[0].clone()]
        )
        .unwrap(),
    );
    update.push(
        term!(
            mk_op("NEQ"),
            [tmp13.clone()],
            [x1[0].clone(), x2[0].clone()]
        )
        .unwrap(),
    );
    update.push(term!(mk_op("OR"), [tmp14.clone()], [tmp12.clone(), tmp13.clone()]).unwrap());
    update.push(
        term!(
            mk_op("AND"),
            [cond2.clone()],
            [tmp14.clone(), tmp11.clone()]
        )
        .unwrap(),
    );

    let in_cs = Wire::new(stype);
    let cond3 = Wire::new("bool");
    update.push(term!(mk_op("CONST(inCS)"), [in_cs.clone()]).unwrap());
    update.push(
        term!(
            mk_op("EQ"),
            [cond3.clone()],
            [in_cs.clone(), pc1[0].clone()]
        )
        .unwrap(),
    );

    let const_true = Wire::new("bool");
    update.push(term!(mk_op("CONST(true)"), [const_true.clone()]).unwrap());

    update.push(
        term!(
            mk_op("CASE"),
            [pc1[1].clone(), x1[1].clone()],
            [
                cond1,
                req_cs,
                x2[0].clone(),
                cond2,
                in_cs.clone(),
                x1[0].clone(),
                cond3,
                out_cs.clone(),
                x1[0].clone(),
                const_true,
                pc1[0].clone(),
                x1[0].clone(),
            ]
        )
        .unwrap(),
    );

    let obs = Interface::from_iter([pc1, x1, pc2, x2]);
    Module::sequential(obs, init, update)
}

fn example_tiny1(
    external: [Wire<&'static str>; 2],
    interface: [Wire<&'static str>; 2],
    wait: bool,
) -> Result<Module, String> {
    let private = [Wire::new("Tny"), Wire::new("Tny")];
    let temp = Wire::new("Tny");

    let cons = Term::constant(mk_op("CONST"), [temp.clone()]).unwrap();

    let update = if wait {
        Term::function(
            mk_op("AWAIT"),
            [interface[1].clone(), private[1].clone()],
            [external[1].clone(), private[0].clone(), temp],
        )
        .unwrap()
    } else {
        Term::function(
            mk_op("SEQ"),
            [interface[1].clone(), private[1].clone()],
            [external[0].clone(), private[0].clone(), temp],
        )
        .unwrap()
    };

    let init = Term::constant(mk_op("INIT"), [interface[1].clone(), private[1].clone()]).unwrap();

    let obs = Interface::from_iter([external, interface]);
    let prvt = Interface::from_iter([private]);

    Module::partially_observable_sequential(obs, prvt, [init], [cons, update])
}

#[test]
fn can_instantiate_sequential_module() {
    let _module = example_counter().unwrap();
    //print!("{}", _module);
}

#[test]
fn can_instantiate_partially_observable_module() {
    let m = example_counter().unwrap();
    let wires = m.obs().clone();
    let mut obs: Vec<[Wire<&'static str>; 2]> = Vec::new();
    let mut prvt: Vec<[Wire<&'static str>; 2]> = Vec::new();
    for [ltc, nxt] in wires {
        if ltc.id() == 0 {
            prvt.push([ltc, nxt]);
        } else {
            obs.push([ltc, nxt]);
        }
    }
    let obs = Interface::from_iter(obs);
    let prvt = Interface::from_iter(prvt);

    let m = Module::partially_observable(obs, prvt, m.atoms().iter().cloned());
    //print!("{}", m);
    assert!(m.is_ok());
}

#[test]
fn cannot_instantiate_external_unobservable_wire() {
    let m = example_counter().unwrap();
    let wires = m.obs().clone();
    let obswire = wires.wire(0, 3).unwrap().clone();
    let mut obs: Vec<[Wire<&'static str>; 2]> = Vec::new();
    let mut prvt: Vec<[Wire<&'static str>; 2]> = Vec::new();
    for [ltc, nxt] in wires {
        if ltc.id() == obswire.id() {
            prvt.push([ltc, nxt]);
        } else {
            obs.push([ltc, nxt]);
        }
    }
    let obs = Interface::from_iter(obs);
    let prvt = Interface::from_iter(prvt);

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

#[test]
fn module_write_all_ctrl() {
    let x = Wire::new("real");
    let xn = Wire::new("real");
    let y = Wire::new("real");
    let yn = Wire::new("real");

    let x0 = Wire::new("real");
    let xn0 = Wire::new("real");

    let update: Vec<Term<Ops>> = [term!(mk_op("ID"), [xn.clone()], [x.clone()]).unwrap()].to_vec();

    let obs = Interface::from_iter([
        [x.clone(), xn.clone()],
        [y.clone(), yn.clone()],
        [x0, xn0.clone()],
    ]);

    let m = Module::sequential(obs.clone(), vec![], update.clone());
    assert!(m.is_err_and(|msg| {
        msg.contains("Controlled wire") && msg.contains("is not written in init")
    }));

    let init: Vec<Term<Ops>> = [term!(mk_op("ID"), [xn0.clone()], [xn.clone()]).unwrap()].to_vec();
    let m = Module::sequential(obs.clone(), init, update.clone());
    assert!(m.is_err_and(|msg| {
        msg.contains("Controlled wire") && msg.contains("is not written in init")
    }));

    let init: Vec<Term<Ops>> = [
        term!(mk_op("ID"), [xn.clone()], [xn0.clone()]).unwrap(),
        term!(mk_op("ID"), [yn.clone()], [xn0]).unwrap(),
    ]
    .to_vec();

    let m = Module::sequential(obs.clone(), init.clone(), update);
    assert!(m.is_err_and(|msg| {
        msg.contains("Controlled wire") && msg.contains("is not written in update")
    }));

    let update: Vec<Term<Ops>> = [
        term!(mk_op("ID"), [xn.clone()], [x.clone()]).unwrap(),
        term!(mk_op("ID"), [yn], [y.clone()]).unwrap(),
    ]
    .to_vec();

    let m = Module::sequential(obs.clone(), init, update);
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
    let x = [Wire::new("Tny"), Wire::new("Tny")];
    let y = [Wire::new("Tny"), Wire::new("Tny")];
    let m = example_tiny1(x, y, true).unwrap();
    assert!(m.is_open());
}

#[test]
fn can_instantiate_example_tiny1_2301() {
    let x = [Wire::new("Tny"), Wire::new("Tny")];
    let y = [Wire::new("Tny"), Wire::new("Tny")];
    let m = example_tiny1(y, x, true).unwrap();
    assert!(m.is_open());
}
#[test]
fn can_compose_example_tiny1() {
    let x = [Wire::new("Tny"), Wire::new("Tny")];
    let y = [Wire::new("Tny"), Wire::new("Tny")];
    let m1 = example_tiny1(x.clone(), y.clone(), false).unwrap();
    let m2 = example_tiny1(y, x, false).unwrap();

    let m3 = Module::parallel([m1, m2]);
    assert!(m3.is_ok());
}

#[test]
fn cannot_compose_example_tiny1_with_cyclic_await() {
    let x = [Wire::new("Tny"), Wire::new("Tny")];
    let y = [Wire::new("Tny"), Wire::new("Tny")];
    let m1 = example_tiny1(x.clone(), y.clone(), true).unwrap();
    let m2 = example_tiny1(y, x, true).unwrap();

    let m3 = Module::parallel([m1, m2]);
    assert!(m3.is_err());
}

#[test]
fn can_compose_example_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let x = [Wire::new("Tny"), Wire::new("Tny")];
    let y = [Wire::new("Tny"), Wire::new("Tny")];
    let m1 = example_tiny1(x.clone(), y.clone(), true).unwrap();
    let m2 = example_tiny1(y, x, false).unwrap();

    let m3 = Module::parallel([m1, m2]);
    assert!(m3.is_ok());
}

#[test]
fn can_compose_three_tiny1_without_cyclic_await_and_overlapping_prvt() {
    let x = [Wire::new("Tny"), Wire::new("Tny")];
    let y = [Wire::new("Tny"), Wire::new("Tny")];
    let z = [Wire::new("Tny"), Wire::new("Tny")];
    let m1 = example_tiny1(x.clone(), y.clone(), true).unwrap();
    let m2 = example_tiny1(y.clone(), x.clone(), false).unwrap();
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
    let x = Wire::new("real");
    let xn = Wire::new("real");
    let y = Wire::new("real");
    let yn = Wire::new("real");
    let z = Wire::new("real");
    let zn = Wire::new("real");

    let assign: Vec<Term<Ops>> = [term!(mk_op("ID"), [xn.clone()], [yn.clone()]).unwrap()].to_vec();
    let obs = Interface::from_iter([[x.clone(), xn.clone()], [y.clone(), yn.clone()]]);
    let m1 = Module::combinatorial(obs.clone(), assign.clone()).unwrap();

    let assign: Vec<Term<Ops>> = [term!(mk_op("ID"), [zn.clone()], [xn.clone()]).unwrap()].to_vec();
    let obs = Interface::from_iter([[x.clone(), xn.clone()], [z.clone(), zn.clone()]]);
    let m2 = Module::combinatorial(obs.clone(), assign.clone()).unwrap();

    Module::parallel([m1, m2]).unwrap();
}

#[test]
fn compose_seq_2() {
    let (x, xn) = (Wire::new("real"), Wire::new("real"));
    let (y, yn) = (Wire::new("real"), Wire::new("real"));
    let (z, zn) = (Wire::new("real"), Wire::new("real"));
    let (y0, y0n) = (Wire::new("real"), Wire::new("real"));
    let (z0, z0n) = (Wire::new("real"), Wire::new("real"));
    let (inv, invn) = (Wire::new("real"), Wire::new("real"));

    // class Module(smt.Module):
    //     def init(self, extl) -> None:
    //         y0, z0 = extl
    //         return Int(0), nxt(y0), nxt(z0)  # = x, y, z
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
        term!(mk_op("Const(0)"), [xn.clone()]).unwrap(),
        term!(mk_op("Id"), [yn.clone()], [y0n.clone()]).unwrap(),
        term!(mk_op("Id"), [zn.clone()], [z0n.clone()]).unwrap(),
    ]
    .to_vec();

    let tmps = [
        Wire::new("real"),
        Wire::new("real"),
        Wire::new("real"),
        Wire::new("real"),
        Wire::new("real"),
        Wire::new("real"),
    ];
    let update: Vec<Term<Ops>> = [
        term!(mk_op("Lt"), [tmps[0].clone()], [x.clone(), y.clone()]).unwrap(),
        term!(mk_op("Lt"), [tmps[1].clone()], [x.clone(), z.clone()]).unwrap(),
        term!(
            mk_op("Or"),
            [tmps[2].clone()],
            [tmps[0].clone(), tmps[1].clone()]
        )
        .unwrap(),
        term!(mk_op("Const(0)"), [tmps[3].clone()]).unwrap(),
        term!(mk_op("Const(1)"), [tmps[4].clone()]).unwrap(),
        term!(
            mk_op("Add"),
            [tmps[5].clone()],
            [x.clone(), tmps[4].clone()]
        )
        .unwrap(),
        term!(
            mk_op("Ite"),
            [xn.clone()],
            [tmps[2].clone(), tmps[5].clone(), tmps[3].clone()]
        )
        .unwrap(),
        term!(mk_op("Id"), [yn.clone()], [y.clone()]).unwrap(),
        term!(mk_op("Id"), [zn.clone()], [z.clone()]).unwrap(),
    ]
    .to_vec();
    let obs = Interface::from_iter([
        [x.clone(), xn.clone()],
        [y.clone(), yn.clone()],
        [z.clone(), zn.clone()],
        [y0.clone(), y0n.clone()],
        [z0.clone(), z0n.clone()],
    ]);
    let m1 = Module::sequential(obs.clone(), init, update).unwrap();

    //
    // class Inv(smt.Module):
    //     def init(self, extl) -> None:
    //         x, y, z = extl
    //         return Or(nxt(x) <= nxt(y), nxt(x) <= nxt(z))
    //
    //     def update(self, inv, extl) -> None:
    //         x, y, z = extl
    //         return Or(nxt(x) <= nxt(y), nxt(x) <= nxt(z))
    let tmps = [Wire::new("real"), Wire::new("real"), Wire::new("real")];
    let assign: Vec<Term<Ops>> = [
        term!(mk_op("Le"), [tmps[0].clone()], [xn.clone(), yn.clone()]).unwrap(),
        term!(mk_op("Le"), [tmps[1].clone()], [xn.clone(), zn.clone()]).unwrap(),
        term!(
            mk_op("Or"),
            [tmps[2].clone()],
            [tmps[0].clone(), tmps[1].clone()]
        )
        .unwrap(),
        term!(mk_op("Id"), [invn.clone()], [tmps[2].clone()]).unwrap(),
    ]
    .to_vec();

    let obs = Interface::from_iter([
        [x.clone(), xn.clone()],
        [y.clone(), yn.clone()],
        [z.clone(), zn.clone()],
        [inv.clone(), invn.clone()],
    ]);
    let m2 = Module::combinatorial(obs.clone(), assign.clone()).unwrap();

    Module::parallel([m1.clone(), m2]).unwrap();

    // try to use a `sequential_observable` ctor instead of combinatorial
    let m2 = Module::sequential(obs.clone(), assign.clone(), assign).unwrap();
    let _m = Module::parallel([m1, m2]).unwrap();
    println!("{:?}", _m);
}

#[test]
fn more_controlled_than_external() {
    let x = [Wire::new("A"), Wire::new("A")];
    let y = [Wire::new("B"), Wire::new("B")];
    let z = [Wire::new("C"), Wire::new("C")];

    let init = Term::constant(mk_op("A"), [y[1].clone(), z[1].clone()]).unwrap();
    let update = Term::function(
        mk_op("A"),
        [y[1].clone(), z[1].clone()],
        [y[0].clone(), z[0].clone()],
    )
    .unwrap();

    let m = Module::partially_observable_sequential([x], [y, z], [init], [update]);
    assert!(m.is_ok());
}

#[test]
fn module_with_invalid_read() {
    let x = [Wire::new("A"), Wire::new("A")];
    let y = [Wire::new("B"), Wire::new("B")];
    let z = [Wire::new("C"), Wire::new("C")];
    let p = Wire::new("P");

    let init = Term::function(
        mk_op("i"),
        [y[1].clone(), z[1].clone()],
        [p.clone(), x[1].clone()],
    )
    .unwrap();
    let update = Term::function(
        mk_op("u"),
        [y[1].clone(), z[1].clone()],
        [p.clone(), x[1].clone(), y[0].clone(), z[0].clone()],
    )
    .unwrap();

    let m = Module::partially_observable_sequential([x], [y, z], [init], [update]);
    println!("{:?}", m);
    assert!(m.is_err());
}

#[allow(unused)]
#[derive(Clone)]
struct SeqOps(&'static str);
#[allow(unused)]
struct DifOps(&'static str);

impl Theory for SeqOps {
    type Sort = &'static str;
    const NAME: &'static str = "SeqOps";
    fn check<R, W, S, E: fmt::Display>(&self, _read: R, _write: W) -> Result<(), String>
    where
        S: TryInto<Self::Sort, Error = E>,
        R: IntoIterator<Item = S>,
        W: IntoIterator<Item = S>,
    {
        Ok(())
    }
}

impl Theory for DifOps {
    type Sort = &'static str;
    const NAME: &'static str = "DifOps";
    fn check<R, W, S, E: fmt::Display>(&self, _read: R, _write: W) -> Result<(), String>
    where
        S: TryInto<Self::Sort, Error = E>,
        R: IntoIterator<Item = S>,
        W: IntoIterator<Item = S>,
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
    let x = (Wire::new("A"), Wire::new("A"));
    let y = (Wire::new("B"), Wire::new("B"));
    let z = (Wire::new("C"), Wire::new("C"));

    let init = Term::constant(SeqOps::HAVOC, [x.1.clone()]).unwrap();
    let jump = Term::function(SeqOps::SKIP, [x.1.clone()], [x.0.clone()]).unwrap();
    let P = base::Module::sequential([x.clone()], [init], [jump]).unwrap();

    let init = Term::constant(SeqOps::HAVOC, [y.1.clone()]).unwrap();
    let flow = Term::constant(DifOps::ZERO, [y.1.clone()]).unwrap();
    let Q = base::Module::differential([y.clone()], [init], [flow]).unwrap();

    let comb = Term::function(SeqOps("+"), [z.1.clone()], [x.1.clone(), y.1.clone()]).unwrap();
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
/// where `t` is an *input*: the module does not control the pair `[t, dt]`,
/// so it is inferred external and the module is open. The clocks `x` and `y`
/// drift at rates 2 and 1 relative to `t`, expressed by delay terms that
/// *await* the input's derivative `dt`: `dx = dt + dt` and `dy = Id(dt)`.
/// Since the rates are relative to the input clock rather than constant in
/// absolute time, these are drifting clocks and not a linear ODE.
#[test]
fn differential_drifting_clocks() {
    use std::collections::HashSet;
    use theory::lra::{LRA, Sort};

    // Real scalars: 1x1 matrices.
    let scalar = Sort::Real([1, 1]);
    // The input clock and its derivative.
    let t = Wire::new(scalar);
    let dt = Wire::new(scalar);
    // The drifting clocks and their derivatives.
    let x = Wire::new(scalar);
    let y = Wire::new(scalar);
    let dx = Wire::new(scalar);
    let dy = Wire::new(scalar);

    // delay: the derivatives, both driven by the input's derivative `dt`.
    let delay = [
        // dx = 2 dt, synthesised tensor-free as dt + dt
        Term::function(LRA::Add(), [dx.clone()], [dt.clone(), dt.clone()]).unwrap(),
        // dy = 1 dt
        Term::function(LRA::Id(), [dy.clone()], [dt.clone()]).unwrap(),
    ];

    // init: the initial derivatives are left unconstrained (Havoc). Every
    // controlled wire must be written by the init block too.
    let init = [
        Term::constant(LRA::Havoc(), [dx.clone()]).unwrap(),
        Term::constant(LRA::Havoc(), [dy.clone()]).unwrap(),
    ];

    // `[latched, derived]` pairs: the two clocks and the input. The atom
    // writes only `dx` and `dy`, so `[t, dt]` is inferred external.
    let obs = [
        [x.clone(), dx.clone()],
        [y.clone(), dy.clone()],
        [t.clone(), dt.clone()],
    ];

    let module: base::Module<LRA, LRA, LRA> = base::Module::differential(obs, init, delay)
        .expect("differential module should be well-formed");

    // An open module: the input clock is external.
    assert!(module.is_open());
    let extl: HashSet<usize> = module.extl().latched().iter().map(Wire::id).collect();
    assert_eq!(extl, HashSet::from([t.id()]));

    // The clocks are controlled by the module, hence interface wires.
    let intf: HashSet<usize> = module.intf().latched().iter().map(Wire::id).collect();
    assert_eq!(intf, HashSet::from([x.id(), y.id()]));

    // The differential constructor produces exactly one atom.
    assert_eq!(module.atoms().len(), 1);
    let atom = &module.atoms()[0];

    // The atom controls the clock derivatives, reads the clocks themselves
    // (for the implicit skip update), and awaits the input's derivative.
    let ctrl: HashSet<usize> = atom.ctrl().ids().collect();
    assert_eq!(ctrl, HashSet::from([dx.id(), dy.id()]));

    let read: HashSet<usize> = atom.read().ids().collect();
    assert_eq!(read, HashSet::from([x.id(), y.id()]));

    let wait: HashSet<usize> = atom.wait().ids().collect();
    assert_eq!(wait, HashSet::from([dt.id()]));

    // Block sizes: 2 derivative terms, 2 init terms, and the update is the
    // implicit `skip` synthesised by the constructor, one term per
    // controlled wire.
    assert_eq!(atom.delay().len(), 2);
    assert_eq!(atom.init().len(), 2);
    assert_eq!(atom.update().len(), 2);

    // All three variables are observable.
    assert_eq!(module.obs().len(), 3);
}
