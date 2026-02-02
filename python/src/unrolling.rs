use pyo3::prelude::*;

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::types::PyAny;

use base::Interface;

// the context in `toy` crate is generic,
// we'll use it until we have the context in `base`.
use crate::{DType, IType, Wire};

use common::unrolling::ModuleUnrolling;

use crate::context::RustContext;
use crate::module::Module;
use crate::{try_term_iter_cloned, try_wire_iter_cloned};

#[pyclass(frozen)]
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

    fn intf_in(slf: PyRef<'_, Self>) -> PyResult<TransitionInterface> {
        let py = slf.py();
        let transition = slf.into_pyobject(py)?.unbind();
        Ok(TransitionInterface {
            transition,
            interface: TransitionInterfaceType::In,
        })
    }

    fn intf_out(slf: PyRef<'_, Self>) -> PyResult<TransitionInterface> {
        let py = slf.py();
        let transition = slf.into_pyobject(py)?.unbind();
        Ok(TransitionInterface {
            transition,
            interface: TransitionInterfaceType::Out,
        })
    }

    fn intf_env(slf: PyRef<'_, Self>) -> PyResult<TransitionEnvInterface> {
        let py = slf.py();
        let transition = slf.into_pyobject(py)?.unbind();
        Ok(TransitionEnvInterface { transition })
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

#[derive(Clone)]
enum TransitionInterfaceType {
    In,
    Out,
}

#[pyclass(sequence)]
struct TransitionInterface {
    transition: Py<Transition>,
    interface: TransitionInterfaceType,
}

#[pymethods]
impl TransitionInterface {
    fn __getitem__(&self, index: usize) -> PyResult<Wire> {
        let transition = &self.transition.get().0;
        // item.and_then(|i| Some(i.map(Clone::clone).map(Wire::from)))
        //     .ok_or(PyIndexError::new_err("index out of bounds"))
        match self.interface {
            TransitionInterfaceType::In => transition
                .intf_in()
                .entry(index)
                .and_then(|w| Some(Wire::from(w[0].clone())))
                .ok_or(PyIndexError::new_err("index out of bounds")),

            TransitionInterfaceType::Out => transition
                .intf_out()
                .entry(index)
                .and_then(|w| Some(Wire::from(w[0].clone())))
                .ok_or(PyIndexError::new_err("index out of bounds")),
        }
    }

    fn __len__(&self) -> usize {
        let transition = &self.transition.get().0;
        match self.interface {
            TransitionInterfaceType::In => transition.intf_in().len(),
            TransitionInterfaceType::Out => transition.intf_out().len(),
        }
    }

    fn __str__(&self) -> String {
        let transition = &self.transition.get().0;
        match self.interface {
            TransitionInterfaceType::In => transition.intf_in().to_string(),
            TransitionInterfaceType::Out => transition.intf_out().to_string(),
        }
    }
}

#[pyclass(sequence)]
struct TransitionEnvInterface {
    transition: Py<Transition>,
}

#[pymethods]
impl TransitionEnvInterface {
    fn __getitem__(&self, index: usize) -> PyResult<(Wire, Wire)> {
        match &self.transition.get().0.intf_env() {
            None => Err(PyIndexError::new_err("interface is empty")),
            Some(intf) => intf
                .entry(index)
                .map(|w| (w[0].clone().into(), w[1].clone().into()))
                .ok_or_else(|| PyIndexError::new_err("index out of bounds")),
        }
    }

    fn __len__(&self) -> usize {
        self.transition
            .get()
            .0
            .intf_env()
            .map_or(0, |intf| intf.len())
    }

    fn __str__(&self) -> String {
        self.transition
            .get()
            .0
            .intf_env()
            .map_or("()".to_string(), |intf| intf.to_string())
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

    fn init(self_: Bound<'_, Self>, module: &Bound<'_, Module>, ctx: &Bound<'_, RustContext>) {
        let module = module.borrow();
        let mut ctx = ctx.borrow_mut();

        let mut unroll = ModuleUnrolling::<DType, IType>::new(&module.base, &mut ctx.ctx);
        unroll.init_ref(&mut self_.borrow_mut().transitions);
    }

    fn step(self_: Bound<'_, Self>, module: &Bound<'_, Module>, ctx: &Bound<'_, RustContext>) {
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

    fn __iadd__(self_: Bound<'_, Self>, t: &Bound<'_, Transition>) {
        let t = t.borrow();
        let slf = &mut self_.borrow_mut().transitions;
        slf.push(t.0.clone()).unwrap();
    }

    fn __add__(self_: Bound<'_, Self>, t: &Bound<'_, Transition>) -> Self {
        let mut new = Self {
            transitions: self_.borrow().transitions.clone(),
        };
        new.transitions.push(t.borrow().0.clone()).unwrap();
        new
    }

    fn __len__(&self) -> usize {
        self.transitions.len()
    }

    fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    // fn last(slf: Bound<'_, Self>) -> PyResult<TransitionRef> {
    //     let py = slf.py();
    //     let pywt = slf.into_pyobject(py)?.unbind();
    //     if pywt.borrow(py).is_empty() {
    //         return Err(PyIndexError::new_err(
    //             "last called on empty WiredTransitions",
    //         ));
    //     }
    //
    //     let n = pywt.borrow(py).__len__() - 1;
    //     Ok(TransitionRef {
    //         transitions: pywt,
    //         idx: n,
    //     })
    // }

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
