//! Helpers shared by the `atom` and `module` integration tests: the
//! uninterpreted `Ops` theory and the example modules built over it.
#![allow(dead_code)]

use base::term::Term;
use base::var::{Var, X};
use base::wire::Wire;
use std::collections::HashMap;
use std::fmt;
use std::fmt::{Display, Formatter};
use theory::{Combinatorial, Differential, Sequential, Theory};

#[derive(Clone, Debug)]
#[allow(unused)]
pub struct Ops(pub &'static str);

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
    fn havoc(_range: &Self::Sort) -> Self {
        Ops("HAVOC")
    }
}

impl Sequential for Ops {
    fn skip(_range: &Self::Sort) -> Self {
        Ops("SKIP")
    }
}

impl Differential for Ops {
    fn zero(_range: &Self::Sort) -> Self {
        Ops("ZERO")
    }
}

impl Display for Ops {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

pub type Module = base::Module<Ops, Ops, Ops>;
pub type Atom = base::Atom<Ops, Ops, Ops>;

pub fn mk_op(name: &'static str) -> Ops {
    Ops(name)
}

#[macro_export]
macro_rules! term {
    ($itype:expr, $write:expr) => {
        Term::constant($itype, $write)
    };

    ($itype:expr, $write:expr, $read:expr) => {
        Term::function($itype, $write, $read)
    };
}

#[allow(clippy::vec_init_then_push)]
pub fn example_counter() -> Result<(Module, HashMap<Var<&'static str>, String>), String> {
    let x = Var::new("real");
    let y = Var::new("real");
    let z = Var::new("real");
    let y0 = Var::new("real");
    let z0 = Var::new("real");

    let mut init: Vec<Term<Ops>> = Vec::new();

    let tmp1 = Wire::scalar("real");
    init.push(term!(mk_op("ZERO"), [tmp1])?);

    let tmp2 = Wire::scalar("bool");
    let tmp3 = Wire::scalar("bool");
    init.push(term!(mk_op("ID"), [X(x)], [tmp1])?);
    init.push(term!(mk_op("ABS"), [tmp2], [X(y0)])?);
    init.push(term!(mk_op("ID"), [X(y)], [tmp2])?);
    init.push(term!(mk_op("ABS"), [tmp3], [X(z0)])?);
    init.push(term!(mk_op("ID"), [X(z)], [tmp3])?);

    let mut update: Vec<Term<Ops>> = Vec::new();

    let tmp4 = Wire::scalar("bool");
    let tmp5 = Wire::scalar("bool");
    let tmp6 = Wire::scalar("bool");
    update.push(term!(mk_op("ZERO"), [tmp1])?);
    update.push(term!(mk_op("LEQ"), [tmp4], [x, y])?);
    update.push(term!(mk_op("LEQ"), [tmp5], [x, z])?);
    update.push(term!(mk_op("OR"), [tmp6], [tmp4, tmp5])?);

    let tmp7 = Wire::scalar("real");
    let tmp8 = Wire::scalar("real");
    update.push(term!(mk_op("ONE"), [tmp7])?);
    update.push(term!(mk_op("ADD"), [tmp8], [*x, tmp7])?);

    update.push(term!(mk_op("ITE"), [X(x)], [tmp6, tmp8, tmp1])?);
    update.push(term!(mk_op("ID"), [X(y)], [y0])?);
    update.push(term!(mk_op("ID"), [X(z)], [z0])?);

    let obs = [x, y, z, y0, z0];

    let module = Module::sequential(&obs, init, update)?;
    // every module variable needs a name: the display indexes them all
    let varnames = [(x, "x"), (y, "y"), (z, "z"), (y0, "y0"), (z0, "z0")]
        .into_iter()
        .map(|(var, name)| (var, name.to_string()))
        .collect();

    Ok((module, varnames))
}

#[allow(clippy::vec_init_then_push)]
pub fn example_peterson1() -> Result<Module, String> {
    let stype = "{outCS, reqCS, inCS}";
    let pc1 = Var::new(stype);
    let x1 = Var::new("bool");
    let pc2 = Var::new(stype);
    let x2 = Var::new("bool");

    let mut init: Vec<Term<Ops>> = Vec::new();
    init.push(term!(mk_op("CONST(outCS)"), [X(pc1)]).unwrap());
    init.push(term!(mk_op("CONST(true)"), [X(x1)]).unwrap());

    let mut update: Vec<Term<Ops>> = Vec::new();
    let out_cs = Wire::scalar(stype);
    let cond1 = Wire::scalar("bool");
    update.push(term!(mk_op("CONST(outCS)"), [out_cs]).unwrap());
    update.push(term!(mk_op("EQ"), [cond1], [out_cs, *pc1]).unwrap());

    let req_cs = Wire::scalar(stype);
    let cond2 = Wire::scalar("bool");
    update.push(term!(mk_op("CONST(reqCS)"), [req_cs]).unwrap());
    let tmp11 = Wire::scalar("bool");
    update.push(term!(mk_op("EQ"), [tmp11], [req_cs, *pc1]).unwrap());

    let tmp12 = Wire::scalar("bool");
    let tmp13 = Wire::scalar("bool");
    let tmp14 = Wire::scalar("bool");
    update.push(term!(mk_op("EQ"), [tmp12], [out_cs, *pc2]).unwrap());
    update.push(term!(mk_op("NEQ"), [tmp13], [x1, x2]).unwrap());
    update.push(term!(mk_op("OR"), [tmp14], [tmp12, tmp13]).unwrap());
    update.push(term!(mk_op("AND"), [cond2], [tmp14, tmp11]).unwrap());

    let in_cs = Wire::scalar(stype);
    let cond3 = Wire::scalar("bool");
    update.push(term!(mk_op("CONST(inCS)"), [in_cs]).unwrap());
    update.push(term!(mk_op("EQ"), [cond3], [in_cs, *pc1]).unwrap());

    let const_true = Wire::scalar("bool");
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
    Module::sequential(&obs, init, update)
}

pub fn example_tiny1(
    external: Var<&'static str>,
    interface: Var<&'static str>,
    wait: bool,
) -> Result<Module, String> {
    let private = Var::new("Tny");
    let temp = Wire::scalar("Tny");

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
    let _obs = [external, interface];
    let prvt = [private];

    let atom = Atom::sequential(&vars, [init], [cons, update])?;
    Module::partially_observable([atom], |v| prvt.contains(v))
}
