use crate::*;
use pyo3::Py;
use pyo3::exceptions::{PyException, PyIndexError, PyTypeError};
use pyo3::types::{PyDict, PyTuple};
use theory::any::Any;

#[pyclass(subclass, frozen)]
#[derive(Debug)]
pub(crate) struct Module {
    pub(crate) base: base::Module<Any, Any, Any>,
}

#[pymethods]
impl Module {
    #[new]
    #[pyo3(signature = (*args, init = None, delay = None, update = None, assign = None, obs = None, ctrl = None, extl = None, intf = None, prvt = None, **_kwargs)
    )]
    fn new(
        args: &Bound<'_, PyTuple>,
        init: Option<&Bound<'_, PyAny>>,
        delay: Option<&Bound<'_, PyAny>>,
        update: Option<&Bound<'_, PyAny>>,
        assign: Option<&Bound<'_, PyAny>>,
        obs: Option<&Bound<'_, PyAny>>,
        ctrl: Option<&Bound<'_, PyAny>>,
        extl: Option<&Bound<'_, PyAny>>,
        intf: Option<&Bound<'_, PyAny>>,
        prvt: Option<&Bound<'_, PyAny>>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if args.len() > 0 {
            if init.is_some()
                || delay.is_some()
                || update.is_some()
                || assign.is_some()
                || obs.is_some()
                || ctrl.is_some()
                || extl.is_some()
                || intf.is_some()
                || prvt.is_some()
                || _kwargs.is_some()
            {
                return Err(PyTypeError::new_err(
                    "positional Module arguments cannot be combined with keyword arguments",
                ));
            }
            return Self::parallel(args);
        }

        match (init, delay, update, assign) {
            (Some(init), None, Some(update), None) => {
                Self::sequential(init, update, obs, ctrl, extl, intf, prvt)
            }
            (None, None, None, Some(assign)) => Self::combinatorial(assign, obs, extl, intf),
            (Some(init), Some(delay), None, None) => {
                Self::differential(init, delay, obs, ctrl, extl, intf, prvt)
            }
            _ => Err(PyTypeError::new_err("unsupported wires declaration")),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (init, update, obs = None, *, ctrl = None, extl = None, intf = None, prvt = None)
    )]
    fn sequential(
        init: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        obs: Option<&Bound<'_, PyAny>>,
        ctrl: Option<&Bound<'_, PyAny>>,
        extl: Option<&Bound<'_, PyAny>>,
        intf: Option<&Bound<'_, PyAny>>,
        prvt: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let update = try_term_iter_cloned(&update)?;

        let module = match (obs, ctrl, extl, intf, prvt) {
            (Some(obs), None, None, None, Some(prvt)) => {
                let obs = try_wire2_iter_cloned(obs)?;
                let prvt = try_wire2_iter_cloned(prvt)?;
                base::Module::partially_observable_sequential(obs, prvt, init, update)
            }
            (Some(wires), None, None, None, None) => {
                let obs = try_wire2_iter_cloned(wires)?;
                base::Module::sequential(obs, init, update)
            }
            (None, Some(_state), Some(_input), Some(_output), None) => {
                todo!() //moore
            }
            (None, None, Some(_extl), Some(_intf), Some(_prvt)) => {
                todo!() //sequential declared
            }
            (Some(_obs), Some(_ctrl), Some(_extl), Some(_intf), Some(_prvt)) => {
                todo!() //sequential fully declared
            }
            (None, None, None, Some(_intf), Some(_prvt)) => {
                todo!() //sequential closed
            }
            _ => return Err(PyTypeError::new_err("unsupported wires declaration")),
        };

        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (assign, obs = None, *, extl = None, intf = None))]
    fn combinatorial(
        assign: &Bound<'_, PyAny>,
        obs: Option<&Bound<'_, PyAny>>,
        extl: Option<&Bound<'_, PyAny>>,
        intf: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let assign = try_term_iter_cloned(&assign)?;

        let module = match (obs, extl, intf) {
            (Some(obs), None, None) => {
                let obs = try_wire2_iter_cloned(obs)?;
                base::Module::combinatorial(obs, assign)
            }
            (None, Some(_extl), Some(_intf)) => {
                todo!() // combinatorial declared
            }
            (None, None, Some(_intf)) => {
                todo!() // constant
            }
            _ => return Err(PyTypeError::new_err("unsupported wires declaration")),
        };

        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (init, flow, obs = None, *, ctrl = None, extl = None, intf = None, prvt = None))]
    fn differential(
        init: &Bound<'_, PyAny>,
        flow: &Bound<'_, PyAny>,
        obs: Option<&Bound<'_, PyAny>>,
        ctrl: Option<&Bound<'_, PyAny>>,
        extl: Option<&Bound<'_, PyAny>>,
        intf: Option<&Bound<'_, PyAny>>,
        prvt: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let flow = try_term_iter_cloned(&flow)?;

        let module = match (obs, ctrl, extl, intf, prvt) {
            (Some(obs), None, None, None, None) => {
                let obs = try_wire2_iter_cloned(obs)?;
                base::Module::differential(obs, init, flow)
            }
            _ => return Err(PyTypeError::new_err("unsupported wires declaration")),
        };

        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (*modules))]
    fn parallel(modules: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let modules = try_iter_borrow::<Self>(modules)?;
        // TODO: make base take result iterator to avoid unwrap
        let modules = modules.into_iter().map(Result::unwrap);
        let modules = modules.map(|r| r.base.clone());

        match base::Module::parallel(modules) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[getter]
    fn extl(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Extl)
    }

    #[getter]
    fn intf(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Intf)
    }

    #[getter]
    fn prvt(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Prvt)
    }

    #[getter]
    fn obs(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Obs)
    }

    #[getter]
    fn ctrl(slf: PyRef<'_, Self>) -> PyResult<ModuleInterface> {
        Self::interface(slf, ModuleInterfaceType::Ctrl)
    }

    #[getter]
    fn atoms(slf: PyRef<'_, Self>) -> PyResult<ModuleAtoms> {
        let py = slf.py();
        let module = slf.into_pyobject(py)?.unbind();
        Ok(ModuleAtoms { module })
    }

    fn closed(&self) -> bool {
        self.base.is_closed()
    }

    fn open(&self) -> bool {
        self.base.is_open()
    }

    fn __str__(&self) -> String {
        self.base.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl From<base::Module<Any, Any, Any>> for Module {
    fn from(base: base::Module<Any, Any, Any>) -> Self {
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
    fn base(&self) -> &base::Interface<Sort, 2> {
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
        let item = self.base().entry(index);
        item.and_then(|i| Some(i.map(Clone::clone).map(Wire::from)))
            .ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        self.base().len()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        let other = match try_array2_iter_borrow::<Wire>(other) {
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
