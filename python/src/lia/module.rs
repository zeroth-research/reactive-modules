use super::Wire;
use super::term::Term;
use crate::*;
use pyo3::exceptions::{PyException, PyIndexError, PyTypeError};
use pyo3::types::{PyDict, PyTuple};
use theory::lia;

#[pyclass(subclass, frozen)]
#[derive(Debug)]
pub struct Module {
    pub(crate) base: base::Module<lia::LIA>,
}

fn try_wire2_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = [base::Wire<lia::Type>; 2]>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_array2_iter_borrow::<Wire>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.map(|r| r.base().clone()));
    Ok(seq)
}

fn try_term_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = base::Term<lia::LIA>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Term>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.base().clone());
    Ok(seq)
}

fn try_wire_iter_cloned(
    seq: &Bound<'_, PyAny>,
) -> PyResult<impl Iterator<Item = base::Wire<lia::Type>>> {
    // TODO: make base take result iterator to avoid unwrap
    let seq = try_iter_borrow::<Wire>(seq)?;
    let seq = seq.into_iter().map(Result::unwrap);
    let seq = seq.map(|r| r.base().clone());
    Ok(seq)
}

#[pymethods]
impl Module {
    #[new]
    #[pyo3(signature = (*_args, init = None, update = None, assign = None, obs = None, ctrl = None, extl = None, intf = None, prvt = None, **_kwargs)
    )]
    fn new(
        _args: &Bound<'_, PyTuple>,
        init: Option<&Bound<'_, PyAny>>,
        update: Option<&Bound<'_, PyAny>>,
        assign: Option<&Bound<'_, PyAny>>,
        obs: Option<&Bound<'_, PyAny>>,
        ctrl: Option<&Bound<'_, PyAny>>,
        extl: Option<&Bound<'_, PyAny>>,
        intf: Option<&Bound<'_, PyAny>>,
        prvt: Option<&Bound<'_, PyAny>>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        if _args.len() > 0 {
            if init.is_some()
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
            return Self::parallel(_args);
        }

        match (init, update, assign) {
            (Some(init), Some(update), None) => {
                Self::sequential(init, update, obs, ctrl, extl, intf, prvt)
            }
            (None, None, Some(assign)) => Self::combinatorial(assign, obs, extl, intf),
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
                base::Module::sequential(obs, prvt, init, update)
            }
            (Some(wires), None, None, None, None) => {
                let wires = try_wire2_iter_cloned(wires)?;
                base::Module::sequential_observable(wires, init, update)
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
            Ok(base) => Ok(Module { base }),
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
            Ok(base) => Ok(Module { base }),
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

impl From<base::Module<lia::LIA>> for Module {
    fn from(base: base::Module<lia::LIA>) -> Self {
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
    fn base(&self) -> &base::Interface<lia::Type, 2> {
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
    fn __getitem__(&self, py: Python<'_>, index: usize) -> PyResult<ModuleAtom> {
        if index >= self.__len__() {
            return Err(PyIndexError::new_err(format!(
                "index {} out of bounds",
                index
            )));
        }
        Ok(ModuleAtom {
            module: self.module.clone_ref(py),
            idx: index,
        })
    }

    fn __len__(&self) -> usize {
        let module = &self.module.get().base;
        module.atoms().len()
    }
}

#[pyclass(frozen)]
struct ModuleAtom {
    module: Py<Module>,
    idx: usize,
}

impl ModuleAtom {
    fn base(&self) -> &base::Atom<lia::LIA> {
        &self.module.get().base.atoms()[self.idx]
    }
}

#[pymethods]
impl ModuleAtom {
    fn init(slf: Bound<'_, Self>) -> ModuleAtomBlock {
        ModuleAtomBlock {
            atom: slf.unbind(),
            is_init: true,
        }
    }

    fn update(slf: Bound<'_, Self>) -> ModuleAtomBlock {
        ModuleAtomBlock {
            atom: slf.unbind(),
            is_init: false,
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base())
    }
}

#[pyclass(frozen, sequence)]
struct ModuleAtomBlock {
    atom: Py<ModuleAtom>,
    is_init: bool,
}

impl ModuleAtomBlock {
    fn base(&self) -> &base::Block<lia::LIA> {
        let atom = self.atom.get().base();
        if self.is_init {
            atom.init()
        } else {
            atom.update()
        }
    }
}

impl super::HasTermAt for ModuleAtomBlock {
    fn term_at(&self, idx: usize) -> Option<&base::Term<lia::LIA>> {
        self.base().get(idx)
    }
}

impl super::ReadWriteIntf for ModuleAtomBlock {
    fn interface(&self, is_read: bool) -> &base::Interface<lia::Type> {
        if is_read { self.base().read() } else { self.base().write() }
    }
}

#[pymethods]
impl ModuleAtomBlock {
    fn read(slf: Bound<'_, Self>) -> ModuleAtomInterface {
        ModuleAtomInterface(super::ReadWriteInterface { owner: slf.unbind(), is_read: true })
    }

    fn write(slf: Bound<'_, Self>) -> ModuleAtomInterface {
        ModuleAtomInterface(super::ReadWriteInterface { owner: slf.unbind(), is_read: false })
    }

    fn __getitem__(slf: Bound<'_, Self>, index: usize) -> PyResult<ModuleTerm> {
        if slf.get().base().get(index).is_none() {
            return Err(PyIndexError::new_err("index out of bounds"));
        }
        Ok(ModuleTerm(super::TermAt { owner: slf.unbind(), idx: index }))
    }

    fn __len__(&self) -> usize {
        self.base().len()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.base())
    }
}

// Accessor for the read/write interface of an atom block
#[pyclass(frozen, sequence)]
struct ModuleAtomInterface(super::ReadWriteInterface<ModuleAtomBlock>);

#[pymethods]
impl ModuleAtomInterface {
    fn __getitem__(&self, index: usize) -> PyResult<Wire> {
        self.0.get_item(index)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __str__(&self) -> String {
        self.0.str()
    }
}

// Accessor for a term in an atom block
#[pyclass(frozen)]
struct ModuleTerm(super::TermAt<ModuleAtomBlock>);

impl super::ReadWriteIntf for ModuleTerm {
    fn interface(&self, is_read: bool) -> &base::Interface<lia::Type> {
        self.0.interface(is_read)
    }
}

#[pymethods]
impl ModuleTerm {
    fn read(slf: Bound<'_, Self>) -> ModuleTermInterface {
        ModuleTermInterface(super::ReadWriteInterface { owner: slf.unbind(), is_read: true })
    }

    fn write(slf: Bound<'_, Self>) -> ModuleTermInterface {
        ModuleTermInterface(super::ReadWriteInterface { owner: slf.unbind(), is_read: false })
    }

    fn __repr__(&self) -> String {
        self.0.repr()
    }
}

// Accessor for the read/write wires of a term
#[pyclass(frozen, sequence)]
struct ModuleTermInterface(super::ReadWriteInterface<ModuleTerm>);

#[pymethods]
impl ModuleTermInterface {
    fn __getitem__(&self, index: usize) -> PyResult<Wire> {
        self.0.get_item(index)
    }

    fn __len__(&self) -> usize {
        self.0.len()
    }

    fn __str__(&self) -> String {
        self.0.str()
    }
}
