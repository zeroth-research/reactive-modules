use super::wire::Wire;
use super::{DType, IType, try_iter_extract};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone)]
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
        let write = try_iter_extract::<Wire>(write)?;
        let read = try_iter_extract::<Wire>(read)?;

        // TODO: make the base take result iterators to avoid unwrap
        let write = write.into_iter().map(Result::unwrap);
        let read = read.into_iter().map(Result::unwrap);

        match base::Term::function(itype, write, read) {
            Ok(base) => Ok(Term { base }),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    pub fn constant(itype: IType, write: &Bound<'_, PyAny>) -> PyResult<Self> {
        let write = try_iter_extract::<Wire>(write)?;

        // TODO: make the base take result iterators to avoid unwrap
        let write = write.into_iter().map(Result::unwrap);

        match base::Term::constant(itype, write) {
            Ok(base) => Ok(Term { base }),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[new]
    #[pyo3(signature = (itype, write, read = None))]
    pub fn new(
        itype: IType,
        write: &Bound<'_, PyAny>,
        read: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        match read {
            Some(read) => Self::function(itype, write, read),
            None => Self::constant(itype, write),
        }
    }
}

impl From<Term> for base::Term<DType, IType> {
    fn from(term: Term) -> Self {
        term.base
    }
}
