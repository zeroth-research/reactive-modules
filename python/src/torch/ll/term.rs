use super::wire::Wire;
use super::{DType, IType};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

#[pyclass]
pub struct Term {
    base: base::Term<DType, IType>,
}

#[pymethods]
impl Term {
    #[staticmethod]
    pub fn function(
        itype: IType,
        write: &Bound<'_, PyAny>,
        read: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        // TODO: make the base take IntoIterator<Result<Wire>>
        let write = write.try_iter()?.map(|item| {
            let item = item.unwrap();
            let wire = item.extract::<Wire>().unwrap();
            wire.base().clone()
        });
        let read = read.try_iter()?.map(|item| {
            let item = item.unwrap();
            let wire = item.extract::<Wire>().unwrap();
            wire.base().clone()
        });

        // TODO: make base errors automatically cast into PyErr
        let base = match (base::Term::function(itype, write, read)) {
            Ok(base) => base,
            Err(msg) => return Err(PyException::new_err(msg)),
        };
        Ok(Term { base })
    }
}
