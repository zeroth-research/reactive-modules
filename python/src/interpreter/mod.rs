mod eval;
mod state;

use state::State;

use crate::module::Module;
use crate::pytensor::PyTensor;
use crate::types::{DType, IType};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::collections::HashMap;
use tch::Tensor;

#[pyclass(unsendable)]
pub struct Interpreter {
    module: base::Module<DType, IType>,
    state: State,
    init_transition: common::transition::Transition<DType, IType>,
    update_transition: common::transition::Transition<DType, IType>,
    initialized: bool,
}

impl Interpreter {
    /// Execute a transition: evaluate all terms, writing results into state.
    fn execute(
        state: &mut State,
        transition: &common::transition::Transition<DType, IType>,
    ) -> Result<(), String> {
        for term in transition.terms() {
            // Collect read values
            let read_vals: Vec<&Tensor> = term
                .read()
                .wires()
                .map(|w| {
                    state
                        .get(w.id())
                        .unwrap_or_else(|| panic!("wire w{} not in state", w.id()))
                })
                .collect();

            let results = eval::eval(term.itype(), &read_vals)?;

            // Write results to state
            for (wire, value) in term.write().wires().zip(results) {
                state.set(wire.id(), value);
            }
        }
        Ok(())
    }

    /// Latch: copy next wire values to latched wire slots.
    fn latch(state: &mut State, module: &base::Module<DType, IType>) {
        for [ltc, nxt] in module.ctrl().iter() {
            if let Some(val) = state.get(nxt.id()) {
                let val = val.shallow_clone();
                state.set(ltc.id(), val);
            }
        }
    }

    /// Load environment inputs into state from a Python dict {wire_id: Tensor}.
    fn load_env_inputs(
        state: &mut State,
        module: &base::Module<DType, IType>,
        env_inputs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        if let Some(inputs) = env_inputs {
            for (key, value) in inputs.iter() {
                let wire_id: usize = key.extract()?;
                let tensor: PyTensor = value.extract()?;
                state.set(wire_id, tensor.tensor);
            }
        } else if !module.extl().is_empty() {
            // If the module has external wires but no inputs provided,
            // that's only an error if we actually need them during execution.
            // We allow it — missing wires will panic at eval time if read.
        }
        Ok(())
    }
}

#[pymethods]
impl Interpreter {
    #[new]
    fn new(module: PyRef<'_, Module>) -> PyResult<Self> {
        let base = module.base.clone();
        let init_transition = common::transition::Transition::from_module_init(&base)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        let update_transition = common::transition::Transition::from_module_update(&base)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        Ok(Self {
            module: base,
            state: State::new(),
            init_transition,
            update_transition,
            initialized: false,
        })
    }

    /// Run the init transition and latch.
    #[pyo3(signature = (env_inputs=None))]
    fn initialize(
        &mut self,
        env_inputs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        Self::load_env_inputs(&mut self.state, &self.module, env_inputs)?;
        Self::execute(&mut self.state, &self.init_transition)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        Self::latch(&mut self.state, &self.module);
        self.initialized = true;
        Ok(())
    }

    /// Run the update transition and latch.
    #[pyo3(signature = (env_inputs=None))]
    fn step(
        &mut self,
        env_inputs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<()> {
        if !self.initialized {
            return Err(PyRuntimeError::new_err(
                "interpreter not initialized; call initialize() first",
            ));
        }
        Self::load_env_inputs(&mut self.state, &self.module, env_inputs)?;
        Self::execute(&mut self.state, &self.update_transition)
            .map_err(|e| PyRuntimeError::new_err(e))?;
        Self::latch(&mut self.state, &self.module);
        Ok(())
    }

    /// Get the tensor value of a wire by its ID.
    fn get(&self, wire_id: usize) -> PyResult<PyTensor> {
        self.state
            .get(wire_id)
            .map(|t| PyTensor {
                tensor: t.shallow_clone(),
            })
            .ok_or_else(|| {
                PyRuntimeError::new_err(format!("wire w{wire_id} not in state"))
            })
    }

    /// Return all wire values as a dict {wire_id: Tensor}.
    fn state_dict(&self) -> HashMap<usize, PyTensor> {
        self.state
            .iter()
            .map(|(&id, t)| {
                (
                    id,
                    PyTensor {
                        tensor: t.shallow_clone(),
                    },
                )
            })
            .collect()
    }
}
