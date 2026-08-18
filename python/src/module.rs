use crate::*;
use pyo3::Py;
use pyo3::exceptions::{PyException, PyIndexError, PyTypeError};
use pyo3::types::{PyDict, PyTuple};
use theory::any::{Combinatorial, Differential, Sequential};

#[pyclass(subclass, frozen)]
#[derive(Debug)]
pub(crate) struct Module {
    pub(crate) base: base::Module<Combinatorial, Sequential, Differential>,
}

#[pymethods]
impl Module {
    #[new]
    #[pyo3(signature = (*args, init = None, delay = None, update = None, obs = None, prvt = None, **_kwargs))]
    fn new(
        args: &Bound<'_, PyTuple>,
        init: Option<&Bound<'_, PyAny>>,
        delay: Option<&Bound<'_, PyAny>>,
        update: Option<&Bound<'_, PyAny>>,
        obs: Option<&Bound<'_, PyAny>>,
        prvt: Option<&Bound<'_, PyAny>>,
        // accepted but unused: subclasses extending Module pass their own
        // keyword arguments through this constructor
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if args.len() > 0 {
            if init.is_some()
                || delay.is_some()
                || update.is_some()
                || obs.is_some()
                || prvt.is_some()
            {
                return Err(PyTypeError::new_err(
                    "positional arguments compose modules in parallel and cannot be combined \
                     with keyword arguments",
                ));
            }
            return Self::parallel(args);
        }

        let Some(obs) = obs else {
            return Err(PyTypeError::new_err(
                "missing `obs`: a module needs some observable variables",
            ));
        };

        match (init, delay, update) {
            (Some(init), None, Some(update)) => Self::sequential(init, update, obs, prvt),
            (Some(init), Some(delay), None) => Self::differential(init, delay, obs, prvt),
            (Some(init), Some(delay), Some(update)) => Self::hybrid(init, update, delay, obs, prvt),
            (None, None, Some(update)) => Self::uninitialized(update, obs, prvt),
            (Some(init), None, None) if prvt.is_none() => Self::constant(init, obs),
            _ => Err(PyTypeError::new_err(
                "invalid combination of blocks: expected `init`+`update` (sequential), \
                 `init`+`delay` (differential), `init`+`update`+`delay` (hybrid), \
                 `update` (uninitialized), or `init` (constant)",
            )),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (init, update, obs, prvt = None))]
    fn sequential(
        init: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        obs: &Bound<'_, PyAny>,
        prvt: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let update = try_term_iter_cloned(&update)?;
        let obs: Vec<_> = try_var_iter_cloned(obs)?.collect();
        let prvt: Vec<_> = match prvt {
            Some(prvt) => try_var_iter_cloned(prvt)?.collect(),
            None => Vec::new(),
        };

        let module: Result<base::Module<Combinatorial, Sequential, Differential>, String> =
            base::Atom::sequential(obs.iter().chain(prvt.iter()), init, update)
                .and_then(|atom| base::Module::partially_observable(obs, prvt, [atom]));

        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    // combinatorial modules are fully observable
    #[staticmethod]
    fn combinatorial(assign: &Bound<'_, PyAny>, obs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let assign = try_term_iter_cloned::<Combinatorial>(&assign)?;
        let obs = try_var_iter_cloned(obs)?;

        match base::Module::combinatorial(obs, assign) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (init, flow, obs, prvt = None))]
    fn differential(
        init: &Bound<'_, PyAny>,
        flow: &Bound<'_, PyAny>,
        obs: &Bound<'_, PyAny>,
        prvt: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let flow = try_term_iter_cloned(&flow)?;
        let obs: Vec<_> = try_var_iter_cloned(obs)?.collect();
        let prvt: Vec<_> = match prvt {
            Some(prvt) => try_var_iter_cloned(prvt)?.collect(),
            None => Vec::new(),
        };

        let module: Result<base::Module<Combinatorial, Sequential, Differential>, String> =
            base::Atom::differential(obs.iter().chain(prvt.iter()), init, flow)
                .and_then(|atom| base::Module::partially_observable(obs, prvt, [atom]));

        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (init, update, delay, obs, prvt = None))]
    fn hybrid(
        init: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
        obs: &Bound<'_, PyAny>,
        prvt: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let update = try_term_iter_cloned(&update)?;
        let delay = try_term_iter_cloned(&delay)?;
        let obs: Vec<_> = try_var_iter_cloned(obs)?.collect();
        let prvt: Vec<_> = match prvt {
            Some(prvt) => try_var_iter_cloned(prvt)?.collect(),
            None => Vec::new(),
        };

        let module: Result<base::Module<Combinatorial, Sequential, Differential>, String> =
            base::Atom::hybrid(obs.iter().chain(prvt.iter()), init, update, delay)
                .and_then(|atom| base::Module::partially_observable(obs, prvt, [atom]));

        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (update, obs, prvt = None))]
    fn uninitialized(
        update: &Bound<'_, PyAny>,
        obs: &Bound<'_, PyAny>,
        prvt: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let update = try_term_iter_cloned(&update)?;
        let obs: Vec<_> = try_var_iter_cloned(obs)?.collect();
        let prvt: Vec<_> = match prvt {
            Some(prvt) => try_var_iter_cloned(prvt)?.collect(),
            None => Vec::new(),
        };

        let module: Result<base::Module<Combinatorial, Sequential, Differential>, String> =
            base::Atom::uninitialized(obs.iter().chain(prvt.iter()), update)
                .and_then(|atom| base::Module::partially_observable(obs, prvt, [atom]));

        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    // constant modules are fully observable
    #[staticmethod]
    fn constant(init: &Bound<'_, PyAny>, obs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let obs = try_var_iter_cloned(obs)?;

        match base::Module::constant(obs, init) {
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

impl From<base::Module<Combinatorial, Sequential, Differential>> for Module {
    fn from(base: base::Module<Combinatorial, Sequential, Differential>) -> Self {
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
    fn base(&self) -> &base::Interface<Sort> {
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
        self.base()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn __getitem__(&self, index: usize) -> PyResult<Var> {
        self.base()
            .iter()
            .nth(index)
            .map(|v| Var::from(*v))
            .ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        self.base().len()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        let other = match try_iter_borrow::<Var>(other) {
            Ok(other) => other,
            Err(_) => return false,
        };

        let mut this = self.base().iter();
        let mut other = other.into_iter();
        loop {
            match (this.next(), other.next()) {
                (Some(this), Some(Ok(other))) => {
                    if this != other.base() {
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
