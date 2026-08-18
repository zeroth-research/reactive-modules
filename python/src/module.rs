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
            if init.is_some() || delay.is_some() || update.is_some() || obs.is_some() {
                return Err(PyTypeError::new_err(
                    "positional arguments take atoms or modules",
                ));
            }

            // a sequence of atoms builds a single module (a process), a
            // sequence of modules composes in parallel
            if args.get_item(0)?.downcast::<Atom>().is_ok() {
                return Self::proc(args, prvt);
            } else {
                return Self::comp(args, prvt);
            }
        }

        let Some(obs) = obs else {
            return Err(PyTypeError::new_err(
                "missing `obs`: a module needs some observable variables",
            ));
        };

        match (init, delay, update) {
            (Some(init), None, Some(update)) => {
                if init.is(update) & prvt.is_none() {
                    Self::combinatorial(init, obs)
                } else {
                    Self::sequential(init, update, obs, prvt)
                }
            }
            (Some(init), Some(delay), None) => Self::differential(init, delay, obs, prvt),
            (Some(init), Some(delay), Some(update)) => Self::hybrid(init, update, delay, obs, prvt),
            (None, None, Some(update)) => Self::jump(update, obs, prvt),
            (None, Some(delay), Some(update)) => Self::uninitialized(update, delay, obs, prvt),
            (Some(init), None, None) if prvt.is_none() => Self::constant(init, obs),
            (None, Some(delay), None) => Self::flow(delay, obs, prvt),
            (None, None, None) if prvt.is_none() => Self::hold(obs),
            // only constant (`init` alone) and hold (no blocks) reach here,
            // both with `prvt` given
            (Some(_), None, None) => Err(PyTypeError::new_err(
                "constant modules (`init` alone) are fully observable and take no `prvt`",
            )),
            (None, None, None) => Err(PyTypeError::new_err(
                "hold modules (no blocks) are fully observable and take no `prvt`",
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

        let module = base::Module::sequential(init, update, obs.iter(), prvt.iter());

        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    // combinatorial modules are fully observable
    #[staticmethod]
    fn combinatorial(assign: &Bound<'_, PyAny>, obs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let assign = try_term_iter_cloned::<Combinatorial>(&assign)?;
        let obs: Vec<_> = try_var_iter_cloned(obs)?.collect();

        match base::Module::combinatorial(assign, obs.iter()) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (init, delay, obs, prvt = None))]
    fn differential(
        init: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
        obs: &Bound<'_, PyAny>,
        prvt: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let flow = try_term_iter_cloned(&delay)?;
        let obs: Vec<_> = try_var_iter_cloned(obs)?.collect();
        let prvt: Vec<_> = match prvt {
            Some(prvt) => try_var_iter_cloned(prvt)?.collect(),
            None => Vec::new(),
        };

        let module = base::Module::differential(init, flow, obs.iter(), prvt.iter());
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

        let module = base::Module::hybrid(init, update, delay, obs.iter(), prvt.iter());

        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (update, obs, prvt = None))]
    fn jump(
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

        let module = base::Module::jump(update, obs.iter(), prvt.iter());

        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (update, delay, obs, prvt = None))]
    fn uninitialized(
        update: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
        obs: &Bound<'_, PyAny>,
        prvt: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let update = try_term_iter_cloned(&update)?;
        let delay = try_term_iter_cloned(&delay)?;
        let obs: Vec<_> = try_var_iter_cloned(obs)?.collect();
        let prvt: Vec<_> = match prvt {
            Some(prvt) => try_var_iter_cloned(prvt)?.collect(),
            None => Vec::new(),
        };

        let module = base::Module::uninitialized(update, delay, obs.iter(), prvt.iter());

        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    // hold modules are fully observable
    #[staticmethod]
    fn hold(obs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let obs: Vec<_> = try_var_iter_cloned(obs)?.collect();

        match base::Module::hold(obs.iter()) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (delay, obs, prvt = None))]
    fn flow(
        delay: &Bound<'_, PyAny>,
        obs: &Bound<'_, PyAny>,
        prvt: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let delay = try_term_iter_cloned(&delay)?;
        let obs: Vec<_> = try_var_iter_cloned(obs)?.collect();
        let prvt: Vec<_> = match prvt {
            Some(prvt) => try_var_iter_cloned(prvt)?.collect(),
            None => Vec::new(),
        };

        let module = base::Module::flow(delay, obs.iter(), prvt.iter());

        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    // constant modules are fully observable
    #[staticmethod]
    fn constant(init: &Bound<'_, PyAny>, obs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let obs: Vec<_> = try_var_iter_cloned(obs)?.collect();

        match base::Module::constant(init, obs.iter()) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    /// Builds a single module (a process) from a sequence of atoms, hiding
    /// the `prvt` variables when given.
    #[staticmethod]
    #[pyo3(signature = (*atoms, prvt = None))]
    fn proc(atoms: &Bound<'_, PyTuple>, prvt: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let atoms: Vec<_> = try_iter_borrow::<Atom>(atoms)?.collect::<PyResult<_>>()?;
        let atoms = atoms.iter().map(|a| a.base().clone());

        let module = if let Some(prvt) = prvt {
            let prvt: Vec<_> = try_var_iter_cloned(prvt)?.collect();
            base::Module::partially_observable(atoms, prvt.iter())
        } else {
            base::Module::observable(atoms)
        };

        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    /// The parallel composition of a sequence of modules: all shared
    /// observable variables are coupled and prvt are hidden when provided.
    #[staticmethod]
    #[pyo3(signature = (*modules, prvt = None))]
    fn comp(modules: &Bound<'_, PyTuple>, prvt: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let modules: Vec<_> = try_iter_borrow::<Self>(modules)?.collect::<PyResult<_>>()?;
        let modules = modules.iter().map(|r| r.base.clone());

        let module = if let Some(prvt) = prvt {
            let prvt: Vec<_> = try_var_iter_cloned(prvt)?.collect();
            base::Module::hiding_composition(modules, prvt.iter())
        } else {
            base::Module::composition(modules)
        };

        match module {
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
