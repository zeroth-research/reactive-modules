use super::atom::Atom;
use super::term::Term;
use super::wire::Wire;
use super::{DType, IType, try_iter_borrow, try_iter_borrow2};
use pyo3::exceptions::{PyException, PyIndexError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyclass(frozen)]
#[derive(Debug)]
pub struct Module {
    base: base::Module<DType, IType>,
}

#[pymethods]
impl Module {
    #[staticmethod]
    #[pyo3(signature = (obs, init, update))]
    fn sequential(
        obs: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let obs = try_iter_borrow2::<Wire>(obs)?;
        let init = try_iter_borrow::<Term>(init)?;
        let update = try_iter_borrow::<Term>(update)?;

        // TODO: make base take result iterator to avoid unwrap
        let obs = obs.into_iter().map(Result::unwrap);
        let init = init.into_iter().map(Result::unwrap);
        let update = update.into_iter().map(Result::unwrap);

        let obs = obs.map(|r| r.map(|r| r.base().clone()));
        let init = init.map(|r| r.base().clone());
        let update = update.map(|r| r.base().clone());

        match base::Module::sequential(obs, init, update) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (obs, assign))]
    fn combinatorial(obs: &Bound<'_, PyAny>, assign: &Bound<'_, PyAny>) -> PyResult<Self> {
        let obs = try_iter_borrow2::<Wire>(obs)?;
        let assign = try_iter_borrow::<Term>(assign)?;

        // TODO: make base take result iterator to avoid unwrap
        let obs = obs.into_iter().map(Result::unwrap);
        let assign = assign.into_iter().map(Result::unwrap);

        let obs = obs.map(|r| r.map(|r| r.base().clone()));
        let assign = assign.map(|r| r.base().clone());

        match base::Module::combinatorial(obs, assign) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*args))]
    fn parallel(args: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let modules = try_iter_borrow::<Self>(args)?;
        // TODO: make base take result iterator to avoid unwrap
        let modules = modules.into_iter().map(Result::unwrap);
        let modules = modules.map(|r| r.base.clone());

        match base::Module::parallel(modules) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    fn extl(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Extl)
    }

    fn intf(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Intf)
    }

    fn prvt(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Prvt)
    }

    fn obs(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Obs)
    }

    fn ctrl(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Ctrl)
    }

    fn atoms(slf: PyRef<'_, Self>) -> PyResult<ModuleAtoms> {
        let py = slf.py();
        let module = slf.into_pyobject(py)?.unbind();
        Ok(ModuleAtoms { module })
    }

    fn __str__(&self) -> String {
        self.base.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl From<base::Module<DType, IType>> for Module {
    fn from(base: base::Module<DType, IType>) -> Self {
        Self { base }
    }
}

impl Module {
    fn interface(
        slf: PyRef<'_, Self>,
        interface: ModuleInterfaceType,
    ) -> PyResult<ModuleInterface> {
        let py = slf.py();
        let module = slf.into_pyobject(py)?.unbind();
        Ok(ModuleInterface { module, interface })
    }
}

#[derive(Clone)]
enum ModuleInterfaceType {
    Extl,
    Intf,
    Prvt,
    Obs,
    Ctrl,
}
#[pyclass(sequence)]
struct ModuleInterface {
    module: Py<Module>,
    interface: ModuleInterfaceType,
}

impl ModuleInterface {
    fn base(&self) -> &base::Interface<DType, 2> {
        let module = &self.module.get().base;
        match self.interface {
            ModuleInterfaceType::Extl => module.extl(),
            ModuleInterfaceType::Intf => module.intf(),
            ModuleInterfaceType::Prvt => module.prvt(),
            ModuleInterfaceType::Obs => module.obs(),
            ModuleInterfaceType::Ctrl => module.ctrl(),
        }
    }
}
#[pymethods]
impl ModuleInterface {
    fn __str__(&self) -> String {
        self.base().to_string()
    }

    fn __getitem__(&self, index: usize) -> PyResult<[Wire; 2]> {
        let interface = self.base();
        if index < interface.len() {
            let entry = interface.entry(index).map(Clone::clone);
            Ok(entry.map(Wire::from))
        } else {
            Err(PyIndexError::new_err("index out of bounds"))
        }
    }

    fn __len__(&self) -> usize {
        self.base().len()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        let other = match try_iter_borrow2::<Wire>(other) {
            Ok(other) => other,
            Err(_) => return false,
        };

        let mut this = self.base().iter();
        let mut other = other.into_iter();
        loop {
            match (this.next(), other.next()) {
                (Some(this), Some(Ok(other))) => {
                    if this.iter().zip(other).any(|(&a, b)| a != b.base()) {
                        return false;
                    }
                }
                (None, None) => return true,
                _ => return false,
            }
        }
    }
}

#[pyclass(sequence)]
struct ModuleAtoms {
    module: Py<Module>,
}

#[pymethods]
impl ModuleAtoms {
    fn __getitem__(&self, index: usize) -> PyResult<Atom> {
        let module = &self.module.get().base;
        let atoms = module.atoms();
        let result = atoms.get(index).map(Clone::clone).map(Into::into);
        result.ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        let module = &self.module.get().base;
        module.atoms().len()
    }
}
