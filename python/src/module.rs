use crate::term::Term;
use crate::wire::Wire;
use crate::{DType, IType, try_iter_extract, try_iter_pair_extract};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Module {
    base: base::Module<DType, IType>,
}

#[pymethods]
impl Module {
    #[staticmethod]
    pub fn sequential(
        obs: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let obs = try_iter_pair_extract::<Wire>(obs)?;
        let init = try_iter_extract::<Term>(init)?;
        let update = try_iter_extract::<Term>(update)?;

        // TODO: make base take result iterator to avoid unwrap
        let obs = obs.into_iter().map(Result::unwrap);
        let init = init.into_iter().map(Result::unwrap);
        let update = update.into_iter().map(Result::unwrap);

        let obs = obs.map(|(l, n)| (l.into(), n.into()));
        let init = init.map(Into::into);
        let update = update.map(Into::into);

        match base::Module::sequential(obs, init, update) {
            Ok(base) => Ok(Module { base }),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    pub fn combinatorial(obs: &Bound<'_, PyAny>, assign: &Bound<'_, PyAny>) -> PyResult<Self> {
        let obs = try_iter_pair_extract::<Wire>(obs)?;
        let assign = try_iter_extract::<Term>(assign)?;

        // TODO: make base take result iterator to avoid unwrap
        let obs = obs.into_iter().map(Result::unwrap);
        let assign = assign.into_iter().map(Result::unwrap);

        let obs = obs.map(|(l, n)| (l.into(), n.into()));
        let assign = assign.map(Into::into);

        match base::Module::combinatorial(obs, assign) {
            Ok(base) => Ok(Module { base }),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*args))]
    pub fn parallel(args: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let modules = try_iter_extract::<Self>(args)?;
        // TODO: make base take result iterator to avoid unwrap
        let modules = modules.into_iter().map(Result::unwrap);
        let modules = modules.map(Into::into);

        match base::Module::parallel(modules) {
            Ok(base) => Ok(Module { base }),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }
}

impl From<Module> for base::Module<DType, IType> {
    fn from(module: Module) -> Self {
        module.base
    }
}
