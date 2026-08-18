use crate::term::{Term, TermInterfaceType};
use crate::var::Var;
use crate::wire::Wire;
use crate::{try_iter_borrow, try_term_iter_cloned, try_var_iter_cloned};
use pyo3::exceptions::{PyException, PyIndexError};
use pyo3::prelude::*;
use theory::any::{Any, Combinatorial, Differential, Sequential};

#[pyclass(frozen)]
pub(crate) struct Atom {
    base: base::Atom<Combinatorial, Sequential, Differential>,
}

impl From<base::Atom<Combinatorial, Sequential, Differential>> for Atom {
    fn from(base: base::Atom<Combinatorial, Sequential, Differential>) -> Self {
        Self { base }
    }
}

#[pymethods]
impl Atom {
    #[new]
    #[pyo3(signature = (vars, init = None, delay = None, update = None))]
    fn new(
        vars: &Bound<'_, PyAny>,
        init: Option<&Bound<'_, PyAny>>,
        delay: Option<&Bound<'_, PyAny>>,
        update: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        match (init, delay, update) {
            (Some(init), None, Some(update)) => Self::sequential(vars, init, update),
            (Some(init), Some(delay), None) => Self::differential(vars, init, delay),
            (Some(init), Some(delay), Some(update)) => Self::hybrid(vars, init, update, delay),
            (None, None, Some(update)) => Self::jump(vars, update),
            (None, Some(delay), Some(update)) => Self::uninitialized(vars, update, delay),
            (Some(init), None, None) => Self::constant(vars, init),
            (None, Some(delay), None) => Self::flow(vars, delay),
            (None, None, None) => Self::hold(vars),
        }
    }

    #[staticmethod]
    fn sequential(
        vars: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let init = try_term_iter_cloned(&init)?;
        let update = try_term_iter_cloned(&update)?;

        match base::Atom::sequential(vars.iter(), init, update) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn differential(
        vars: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let init = try_term_iter_cloned(&init)?;
        let delay = try_term_iter_cloned(&delay)?;

        match base::Atom::differential(vars.iter(), init, delay) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn hybrid(
        vars: &Bound<'_, PyAny>,
        init: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let init = try_term_iter_cloned(&init)?;
        let update = try_term_iter_cloned(&update)?;
        let delay = try_term_iter_cloned(&delay)?;

        match base::Atom::hybrid(vars.iter(), init, update, delay) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn jump(vars: &Bound<'_, PyAny>, update: &Bound<'_, PyAny>) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let update = try_term_iter_cloned(&update)?;

        match base::Atom::jump(vars.iter(), update) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn uninitialized(
        vars: &Bound<'_, PyAny>,
        update: &Bound<'_, PyAny>,
        delay: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let update = try_term_iter_cloned(&update)?;
        let delay = try_term_iter_cloned(&delay)?;

        match base::Atom::uninitialized(vars.iter(), update, delay) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn constant(vars: &Bound<'_, PyAny>, init: &Bound<'_, PyAny>) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let init = try_term_iter_cloned(&init)?;

        match base::Atom::constant(vars.iter(), init) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn hold(vars: &Bound<'_, PyAny>) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();

        match base::Atom::hold(vars.iter()) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn flow(vars: &Bound<'_, PyAny>, delay: &Bound<'_, PyAny>) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let delay = try_term_iter_cloned(&delay)?;

        match base::Atom::flow(vars.iter(), delay) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[staticmethod]
    fn combinatorial(vars: &Bound<'_, PyAny>, assign: &Bound<'_, PyAny>) -> PyResult<Self> {
        let vars: Vec<_> = try_var_iter_cloned(vars)?.collect();
        let assign = try_term_iter_cloned::<Combinatorial>(&assign)?;

        match base::Atom::combinatorial(vars.iter(), assign) {
            Ok(base) => Ok(base.into()),
            Err(msg) => Err(PyException::new_err(msg)),
        }
    }

    #[getter]
    fn read(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Read)
    }

    #[getter]
    fn ctrl(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Ctrl)
    }

    #[getter]
    fn wait(slf: PyRef<'_, Self>) -> PyResult<AtomInterface> {
        Self::interface(slf, AtomInterfaceType::Await)
    }

    #[getter]
    fn init(slf: Bound<'_, Self>) -> AtomBlock {
        AtomBlock {
            atom: slf.unbind(),
            block: BlockType::Init,
        }
    }

    #[getter]
    fn update(slf: Bound<'_, Self>) -> AtomBlock {
        AtomBlock {
            atom: slf.unbind(),
            block: BlockType::Update,
        }
    }

    #[getter]
    fn delay(slf: Bound<'_, Self>) -> AtomBlock {
        AtomBlock {
            atom: slf.unbind(),
            block: BlockType::Delay,
        }
    }

    fn __str__(&self) -> String {
        self.base.to_string()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base)
    }
}

impl Atom {
    pub(crate) fn base(&self) -> &base::Atom<Combinatorial, Sequential, Differential> {
        &self.base
    }

    fn interface(slf: PyRef<'_, Self>, interface: AtomInterfaceType) -> PyResult<AtomInterface> {
        let py = slf.py();
        let atom = slf.into_pyobject(py)?.unbind();
        Ok(AtomInterface { atom, interface })
    }
}

#[derive(Clone)]
enum AtomInterfaceType {
    Read,
    Ctrl,
    Await,
}

#[pyclass(sequence)]
struct AtomInterface {
    atom: Py<Atom>,
    interface: AtomInterfaceType,
}

impl AtomInterface {
    fn base(&self) -> &base::Interface<theory::any::Sort> {
        let atom = &self.atom.get().base;
        match self.interface {
            AtomInterfaceType::Read => atom.read(),
            AtomInterfaceType::Await => atom.wait(),
            AtomInterfaceType::Ctrl => atom.ctrl(),
        }
    }
}
#[pymethods]
impl AtomInterface {
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

    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> bool {
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

#[derive(Clone)]
enum BlockType {
    Init,
    Update,
    Delay,
}

#[pyclass(sequence, frozen)]
pub(crate) struct AtomBlock {
    atom: Py<Atom>,
    block: BlockType,
}

// impl AtomBlock {
//     fn base(&self) -> &base::Block<Sequential> {
//         let atom = &self.atom.get().base;
//         match self.block {
//             BlockType::Init => atom.init(),
//             BlockType::Update => atom.update(),
//             BlockType::Delay => atom.delay(),
//         }
//     }
// }

#[pymethods]
impl AtomBlock {
    // requires display in base - do you need that now?
    // fn __str__(&self) -> String {
    //     self.base().to_string()
    // }

    fn __repr__(&self) -> String {
        let atom = &self.atom.get().base;
        match self.block {
            BlockType::Init => atom.init().to_string(),
            BlockType::Update => atom.update().to_string(),
            BlockType::Delay => atom.delay().to_string(),
        }
    }

    fn read(slf: Bound<'_, Self>) -> AtomBlockInterface {
        AtomBlockInterface {
            block: slf.unbind(),
            interface: TermInterfaceType::Read,
        }
    }

    fn write(slf: Bound<'_, Self>) -> AtomBlockInterface {
        AtomBlockInterface {
            block: slf.unbind(),
            interface: TermInterfaceType::Write,
        }
    }

    fn __getitem__(&self, index: usize) -> PyResult<Term> {
        let atom = &self.atom.get().base;
        let item: Option<base::Term<Any>> = match self.block {
            BlockType::Init => atom.init().get(index).cloned().map(Into::into),
            BlockType::Update => atom.update().get(index).cloned().map(Into::into),
            BlockType::Delay => atom.delay().get(index).cloned().map(Into::into),
        };
        item.map(Into::into)
            .ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        let atom = &self.atom.get().base;
        match self.block {
            BlockType::Init => atom.init().len(),
            BlockType::Update => atom.update().len(),
            BlockType::Delay => atom.delay().len(),
        }
    }
}

#[pyclass(sequence)]
struct AtomBlockInterface {
    block: Py<AtomBlock>,
    interface: TermInterfaceType,
}

impl AtomBlockInterface {
    fn base(&self) -> &[base::Wire<theory::any::Sort>] {
        let atom = &self.block.get().atom.get().base;
        match (&self.block.get().block, &self.interface) {
            (BlockType::Init, TermInterfaceType::Read) => atom.init().read(),
            (BlockType::Init, TermInterfaceType::Write) => atom.init().write(),
            (BlockType::Update, TermInterfaceType::Read) => atom.update().read(),
            (BlockType::Update, TermInterfaceType::Write) => atom.update().write(),
            (BlockType::Delay, TermInterfaceType::Read) => atom.delay().read(),
            (BlockType::Delay, TermInterfaceType::Write) => atom.delay().write(),
        }
    }
}

#[pymethods]
impl AtomBlockInterface {
    fn __str__(&self) -> String {
        self.base()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        let other = match try_iter_borrow::<Wire>(other) {
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

    fn __getitem__(&self, index: usize) -> PyResult<Wire> {
        self.base()
            .get(index)
            .map(|w| w.clone().into())
            .ok_or(PyIndexError::new_err("index out of bounds"))
    }

    fn __len__(&self) -> usize {
        self.base().len()
    }
}
