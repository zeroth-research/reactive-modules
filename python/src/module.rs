use crate::*;
use pyo3::Py;
use pyo3::exceptions::{PyException, PyIndexError, PyTypeError};
use pyo3::types::{PyDict, PyTuple};
use std::borrow::Cow;
use std::collections::HashMap;
use theory::any::{Combinatorial, Differential, Sequential};

#[pyclass(subclass, frozen)]
pub(crate) struct Module {
    pub(crate) base: base::Module<Combinatorial, Sequential, Differential>,
}

#[pymethods]
impl Module {
    #[new]
    #[pyo3(signature = (*args, init = None, delay = None, update = None, vars = None, hide = None, **_kwargs))]
    fn new(
        args: &Bound<'_, PyTuple>,
        init: Option<&Bound<'_, PyAny>>,
        delay: Option<&Bound<'_, PyAny>>,
        update: Option<&Bound<'_, PyAny>>,
        vars: Option<&Bound<'_, PyAny>>,
        hide: Option<&Bound<'_, PyAny>>,
        // accepted but unused: subclasses extending Module pass their own
        // keyword arguments through this constructor
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if args.len() > 0 {
            if init.is_some() || delay.is_some() || update.is_some() || vars.is_some() {
                return Err(PyTypeError::new_err(
                    "positional arguments take atoms or modules",
                ));
            }

            // a sequence of atoms builds a single module (a process), a
            // sequence of modules composes in parallel
            if args.get_item(0)?.downcast::<Atom>().is_ok() {
                return Self::proc(args, hide);
            } else {
                return Self::comp(args, hide);
            }
        }

        let Some(vars) = vars else {
            return Err(PyTypeError::new_err(
                "missing `vars`: a module needs some variables",
            ));
        };

        match (init, delay, update) {
            (Some(init), None, Some(update)) => {
                if init.is(update) & hide.is_none() {
                    Self::combinatorial(vars, init)
                } else {
                    Self::sequential(vars, init, update, hide)
                }
            }
            (Some(init), Some(delay), None) => Self::differential(vars, init, delay, hide),
            (Some(init), Some(delay), Some(update)) => {
                Self::hybrid(vars, init, update, delay, hide)
            }
            (None, None, Some(update)) => Self::jump(vars, update, hide),
            (None, Some(delay), Some(update)) => Self::uninitialized(vars, update, delay, hide),
            (None, Some(delay), None) => Self::flow(vars, delay, hide),
            (Some(init), None, None) if hide.is_none() => Self::constant(vars, init),
            (None, None, None) if hide.is_none() => Self::hold(vars),
            // only constant (`init` alone) and hold (no blocks) reach here,
            // both with `hide` given
            (Some(_), None, None) => Err(PyTypeError::new_err(
                "constant modules (`init` alone) are fully observable and take no `hide`",
            )),
            (None, None, None) => Err(PyTypeError::new_err(
                "hold modules (no blocks) are fully observable and take no `hide`",
            )),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars,init, update, *, hide = None))]
    fn sequential(
        vars: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let update = try_term_iter_cloned(&update)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let hide = Hide::try_from(hide)?;

        let module = base::Module::sequential(vars.iter(), init, update, hide.as_fn());

        hide.err()?;
        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    // combinatorial modules are fully observable
    #[staticmethod]
    fn combinatorial(vars: &Bound<'_, PyAny>, assign: &Bound<'_, PyAny>) -> PyResult<Self> {
        let assign = try_term_iter_cloned::<Combinatorial>(&assign)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();

        match base::Module::combinatorial(vars.iter(), assign) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars, init, delay, *, hide = None))]
    fn differential(
        vars: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let flow = try_term_iter_cloned(&delay)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let hide = Hide::try_from(hide)?;

        let module = base::Module::differential(vars.iter(), init, flow, hide.as_fn());

        hide.err()?;
        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars, init, update, delay, *, hide = None))]
    fn hybrid(
        vars: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let update = try_term_iter_cloned(&update)?;
        let delay = try_term_iter_cloned(&delay)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let hide = Hide::try_from(hide)?;

        let module = base::Module::hybrid(vars.iter(), init, update, delay, hide.as_fn());

        hide.err()?;
        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars, update, *, hide = None))]
    fn jump(
        vars: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let update = try_term_iter_cloned(&update)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let hide = Hide::try_from(hide)?;

        let module = base::Module::jump(vars.iter(), update, hide.as_fn());

        hide.err()?;
        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars, update, delay, *, hide = None))]
    fn uninitialized(
        vars: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let update = try_term_iter_cloned(&update)?;
        let delay = try_term_iter_cloned(&delay)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let hide = Hide::try_from(hide)?;

        let module = base::Module::uninitialized(vars.iter(), update, delay, hide.as_fn());

        hide.err()?;
        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    // hold modules are fully observable
    #[staticmethod]
    fn hold(vars: &Bound<'_, PyAny>) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();

        match base::Module::hold(vars.iter()) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vars, delay, *, hide = None))]
    fn flow(
        vars: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
        hide: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let delay = try_term_iter_cloned(&delay)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let hide = Hide::try_from(hide)?;

        let module = base::Module::flow(vars.iter(), delay, hide.as_fn());

        hide.err()?;
        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    // constant modules are fully observable
    #[staticmethod]
    fn constant(vars: &Bound<'_, PyAny>, init: &Bound<'_, PyAny>) -> PyResult<Self> {
        let init = try_term_iter_cloned(&init)?;
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();

        match base::Module::constant(vars.iter(), init) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    /// Builds a single module (a process) from a sequence of atoms, hiding
    /// the `hide` variables when given.
    #[staticmethod]
    #[pyo3(signature = (*atoms, hide = None))]
    fn proc(atoms: &Bound<'_, PyTuple>, hide: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let atoms: Vec<_> = try_iter_borrow::<Atom>(atoms)?.collect::<PyResult<_>>()?;
        let atoms = atoms.iter().map(|a| a.base().clone());
        let hide = Hide::try_from(hide)?;

        let module = base::Module::partially_observable(atoms, hide.as_fn());

        hide.err()?;
        match module {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    /// The parallel composition of a sequence of modules: all shared
    /// observable variables are coupled and the `hide` variables are hidden
    /// when provided.
    #[staticmethod]
    #[pyo3(signature = (*modules, hide = None))]
    fn comp(modules: &Bound<'_, PyTuple>, hide: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let modules: Vec<_> = try_iter_borrow::<Self>(modules)?.collect::<PyResult<_>>()?;
        let modules = modules.iter().map(|r| r.base.clone());
        let hide = Hide::try_from(hide)?;

        let module = base::Module::hiding_composition(modules, hide.as_fn());

        hide.err()?;
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

    /// Renders the module with variables named by `names`; variables
    /// without an entry print as `#{id}`. Don't rely on the fallback - it may change anytime
    fn show(&self, names: HashMap<Var, String>) -> String {
        self.base
            .with_varnames(|v| match names.get(v) {
                Some(name) => Cow::Borrowed(name.as_str()),
                None => Cow::Owned(format!("#{}", v.id())), //
            })
            .to_string()
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

    fn __contains__(&self, item: &Bound<'_, PyAny>) -> bool {
        if let Ok(var) = item.extract::<PyRef<Var>>() {
            return self.base().contains(var.base());
        }
        //also checking containment of any wire (latched, next, derived)
        if let Ok(wire) = item.extract::<PyRef<Wire>>() {
            return self.base().var(wire.base()).is_some();
        }
        false
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
