use pyo3::prelude::*;

use pyo3::exceptions::PyValueError;
use pyo3::types::PyAny;

use base::Interface;

// the context in `toy` crate is generic,
// we'll use it until we have the context in `base`.
use crate::{DType, IType};

use bmc::unrolling::ModuleUnrolling;

use crate::context::RustContext;
use crate::module::Module;
use crate::{try_term_iter_cloned, try_wire_iter_cloned};

#[pyclass]
pub struct Transition(pub(crate) common::transition::Transition<DType, IType>);

#[pymethods]
impl Transition {
    #[new]
    pub fn new(
        vars_in: &Bound<'_, PyAny>,
        vars_env: &Bound<'_, PyAny>,
        vars_env_new: &Bound<'_, PyAny>,
        vars_out: &Bound<'_, PyAny>,
        terms: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let intf_in =
            Interface::sequence(try_wire_iter_cloned(vars_in)?).map_err(PyValueError::new_err)?;
        let intf_out =
            Interface::sequence(try_wire_iter_cloned(vars_out)?).map_err(PyValueError::new_err)?;
        let tmp_env = Interface::try_from_iter(
            try_wire_iter_cloned(vars_env)?.zip(try_wire_iter_cloned(vars_env_new)?),
        )
        .map_err(PyValueError::new_err)?;

        let intf_env = if tmp_env.is_empty() {
            None
        } else {
            Some(tmp_env)
        };

        let terms = try_term_iter_cloned(&terms)?.collect::<Vec<base::Term<DType, IType>>>();

        Ok(Self(common::transition::Transition::new(
            intf_in, intf_env, intf_out, terms,
        )))
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
pub struct WiredTransitions {
    pub(crate) transitions: common::transition::WiredTransitions<DType, IType>,
}

impl Default for WiredTransitions {
    fn default() -> Self {
        Self::new()
    }
}

#[pymethods]
impl WiredTransitions {
    #[new]
    pub fn new() -> Self {
        Self {
            transitions: common::transition::WiredTransitions::new(),
        }
    }

    fn init(
        self_: Bound<'_, Self>,
        module: &Bound<'_, Module>,
        ctx: &Bound<'_, RustContext>,
    ) {
        let module = module.borrow();
        let mut ctx = ctx.borrow_mut();

        let mut unroll = ModuleUnrolling::<DType, IType>::new(&module.base, &mut ctx.ctx);
        unroll.init_ref(&mut self_.borrow_mut().transitions);
    }

    fn step(
        self_: Bound<'_, Self>,
        module: &Bound<'_, Module>,
        ctx: &Bound<'_, RustContext>,
    ) {
        let module = module.borrow();
        let mut ctx = ctx.borrow_mut();

        let mut unroll = ModuleUnrolling::<DType, IType>::new(&module.base, &mut ctx.ctx);
        unroll.step_ref(&mut self_.borrow_mut().transitions);
    }

    fn wire_transition(
        self_: Bound<'_, Self>,
        t: &Bound<'_, Transition>,
        ctx: &Bound<'_, RustContext>,
    ) {
        let t = t.borrow();
        let slf = &mut self_.borrow_mut().transitions;
        let mut ctx = ctx.borrow_mut();
        slf.wire_transition(&t.0, &mut ctx.ctx).unwrap();
    }

    fn dbg(&self) {
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

    #[cfg(feature = "visual-html")]
    fn to_html(&self, _ctx: &Bound<'_, RustContext>, path: &str) {
        //let ctx: &WrappedContext = &ctx.borrow();
        let _ = visual::html::unrolling::write_to_html(
            &self.transitions,
            path, //Some(&ctx.to_smt_ctx(&self.module)),
        );
    }
}
