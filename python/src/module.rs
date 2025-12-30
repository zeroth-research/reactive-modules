use crate::term::Term;
use crate::wire::Wire;
use crate::{DType, IType, try_iter_borrow, try_iter_borrow2};
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

#[pyclass]
#[derive(Debug)]
pub(crate) struct Module {
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
}

impl From<base::Module<DType, IType>> for Module {
    fn from(base: base::Module<DType, IType>) -> Self {
        Self { base }
    }
}

impl Module {
    fn interface(slf: PyRef<'_, Self>, mitype: ModuleInterfaceType) -> PyResult<ModuleInterface> {
        let py = slf.py();
        Ok(ModuleInterface {
            module: slf.into_pyobject(py)?.unbind(),
            interface: mitype,
        })
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
#[pyclass]
struct ModuleInterface {
    module: Py<Module>,
    interface: ModuleInterfaceType,
}
#[pymethods]
impl ModuleInterface {
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Py<ModuleInterfaceIter>> {
        Py::new(
            py,
            ModuleInterfaceIter {
                module: self.module.clone_ref(py),
                interface: self.interface.clone(),
                index: 0,
            },
        )
    }

    fn __str__(slf: PyRef<'_, Self>) -> String {
        let base_module = &slf.module.borrow(slf.py()).base;
        match slf.interface {
            ModuleInterfaceType::Extl => base_module.extl().to_string(),
            ModuleInterfaceType::Intf => base_module.intf().to_string(),
            ModuleInterfaceType::Prvt => base_module.prvt().to_string(),
            ModuleInterfaceType::Obs => base_module.obs().to_string(),
            ModuleInterfaceType::Ctrl => base_module.ctrl().to_string(),
        }
    }
}

#[pyclass]
struct ModuleInterfaceIter {
    module: Py<Module>,
    interface: ModuleInterfaceType,
    index: usize,
}

#[pymethods]
impl ModuleInterfaceIter {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<[Wire; 2]> {
        let result = {
            let base_module = &slf.module.borrow(slf.py()).base;
            let base_interface = match slf.interface {
                ModuleInterfaceType::Extl => base_module.extl(),
                ModuleInterfaceType::Intf => base_module.intf(),
                ModuleInterfaceType::Prvt => base_module.prvt(),
                ModuleInterfaceType::Obs => base_module.obs(),
                ModuleInterfaceType::Ctrl => base_module.ctrl(),
            };
            (slf.index < base_interface.len()).then(|| {
                base_interface
                    .entry(slf.index)
                    .map(Clone::clone)
                    .map(Wire::from)
            })
        };
        slf.index += 1;
        result
    }
}
