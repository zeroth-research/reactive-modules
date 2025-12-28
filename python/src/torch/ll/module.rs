use super::term::Term;
use super::wire::Wire;
use super::{DType, IType, try_iter_extract, try_iter_pair_extract};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

#[pyclass]
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

        match base::Module::combinatorial(obs, assign) {
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
