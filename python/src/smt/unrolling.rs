use crate::smt::PyVal;
use pyo3::prelude::*;

use pyo3::types::PyList;

use base::Interface;

// the context in `toy` crate is generic,
// we'll use it until we have the context in `base`.
use smt::dtype::DType;
use smt::itype::IType;

use bmc::unrolling::ModuleUnrolling;
use common::transition::{Transition, WiredTransitions};

use super::wrappedcontext::WrappedContext;
use super::wrappedmodule::WrappedModule;

#[pyclass]
pub struct WrappedTransition(pub(crate) Transition<DType, IType>);

fn vars_to_intf(vars: &[Py<PyVal>]) -> Interface<DType> {
    Interface::sequence(vars.iter().map(|val| match val.get() {
        PyVal::Sym(id, ty) => {
            let ty: DType = ty.parse().expect("Failed parsing DType");
            base::Wire::new(*id, ty)
        }
        _ => panic!("Invalid PyVal, expected PyVal::Sym"),
    }))
    .unwrap()
}

#[pymethods]
impl WrappedTransition {
    #[new]
    pub fn new(
        ctx: &Bound<'_, WrappedContext>,
        vars_in: Vec<Py<PyVal>>,
        vars_env: Vec<Py<PyVal>>,
        vars_env_new: Vec<Py<PyVal>>,
        vars_out: Vec<Py<PyVal>>,
        terms: &Bound<'_, PyList>,
    ) -> Self {
        let intf_in = vars_to_intf(&vars_in);
        let intf_out = vars_to_intf(&vars_out);
        let intf_env = if vars_env.is_empty() {
            assert!(vars_env_new.is_empty());
            None
        } else {
            Some(
                Interface::try_from_iter(
                    vars_to_intf(&vars_env)
                        .into_iter()
                        .zip(vars_to_intf(&vars_env_new).into_iter())
                        .map(|(w1, w2)| [w1[0].clone(), w2[0].clone()]),
                )
                .unwrap(),
            )
        };

        let ctx: &mut WrappedContext = &mut ctx.borrow_mut();
        let terms = crate::smt::wterms_to_terms(ctx, terms).unwrap();

        Self(Transition::new(intf_in, intf_env, intf_out, terms))
    }

    pub fn dbg(&self) {
        println!("---------------------------");
        println!("Transition:");
        if let Some(intf_env) = self.0.intf_env() {
            println!("In: {}", self.0.intf_in());
            println!("Env: {}", intf_env);
        } else {
            println!("In: {}", self.0.intf_in());
        }
        println!("---------------------------");
        for term in &self.0.transition {
            println!("{}", term);
        }
        println!("---------------------------");
        println!("Out: {}", self.0.intf_out());
    }
}

#[pyclass]
pub struct WrappedWiredTransitions {
    pub(crate) transitions: WiredTransitions<DType, IType>,
}

impl Default for WrappedWiredTransitions {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl WrappedWiredTransitions {
    #[new]
    pub fn new() -> Self {
        Self {
            transitions: WiredTransitions::new(),
        }
    }

    pub fn init(
        self_: Bound<'_, Self>,
        module: &Bound<'_, WrappedModule>,
        ctx: &Bound<'_, WrappedContext>,
    ) {
        let module = module.borrow();
        let mut ctx = ctx.borrow_mut();

        let mut unroll = ModuleUnrolling::<DType, IType>::new(&module.module, &mut ctx.ctx);
        unroll.init_ref(&mut self_.borrow_mut().transitions);
    }

    pub fn step(
        self_: Bound<'_, Self>,
        module: &Bound<'_, WrappedModule>,
        ctx: &Bound<'_, WrappedContext>,
    ) {
        let module = module.borrow();
        let mut ctx = ctx.borrow_mut();

        let mut unroll = ModuleUnrolling::<DType, IType>::new(&module.module, &mut ctx.ctx);
        unroll.step_ref(&mut self_.borrow_mut().transitions);
    }

    pub fn wire_transition(
        self_: Bound<'_, Self>,
        t: &Bound<'_, WrappedTransition>,
        ctx: &Bound<'_, WrappedContext>,
    ) {
        let t = t.borrow();
        let slf = &mut self_.borrow_mut().transitions;
        let mut ctx = ctx.borrow_mut();
        slf.wire_transition(&t.0, &mut ctx.ctx).unwrap();
    }

    pub fn dbg(&self) {
        for transition in &self.transitions {
            println!("---------------------------");
            if let Some(intf_env) = transition.intf_env() {
                println!("In: {}", transition.intf_in());
                println!("Env: {}", intf_env);
            } else {
                println!("In: {}", transition.intf_in());
            }
            println!("---------------------------");
            for term in transition {
                println!("{}", term);
            }
            println!("---------------------------");
            println!("Out: {}", transition.intf_out());
        }
    }
}
