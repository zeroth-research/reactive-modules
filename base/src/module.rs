use crate::atom::Atom;
use crate::term::Term;
use crate::wire::{Interface, Wire};
use crate::{Error, topological_order};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fmt::Debug;
use theory::Theory;

/// This data structure corresponds to the module of reactive modules.
#[derive(Debug, Clone)]
pub struct Module<T: Theory> {
    /// Correspond to the wires of the module divided by visibility
    /// ```text
    ///     *====================*
    ///     | extl | intf | prvt |
    ///     *--------------------*
    ///     |     obs     | prvt |
    ///     *--------------------*
    ///     | extl |    ctrl     |
    ///     *====================*
    /// ```
    ///  Wires are organised in pairs of identical twins where
    ///  - 0: latched wires
    ///  - 1: next wires
    extl: Interface<T::Sort, 2>,
    intf: Interface<T::Sort, 2>,
    prvt: Interface<T::Sort, 2>,
    obs: Interface<T::Sort, 2>,
    ctrl: Interface<T::Sort, 2>,
    temp: Interface<T::Sort>,

    /// The atoms of this module.
    /// The atoms must be stored in a *consistent* linear order
    /// as defined in the reactive modules paper.
    atoms: Vec<Atom<T>>,
}

impl<T: Theory> Module<T> {
    pub fn atoms(&self) -> &[Atom<T>] {
        &self.atoms
    }

    pub fn extl(&self) -> &Interface<T::Sort, 2> {
        &self.extl
    }

    pub fn intf(&self) -> &Interface<T::Sort, 2> {
        &self.intf
    }

    pub fn prvt(&self) -> &Interface<T::Sort, 2> {
        &self.prvt
    }

    pub fn ctrl(&self) -> &Interface<T::Sort, 2> {
        &self.ctrl
    }

    pub fn obs(&self) -> &Interface<T::Sort, 2> {
        &self.obs
    }

    pub fn is_closed(&self) -> bool {
        self.extl.is_empty()
    }

    pub fn is_open(&self) -> bool {
        !self.extl.is_empty()
    }

    pub fn temp(&self) -> impl Iterator<Item = &Wire<T::Sort>> {
        self.temp.wires()
    }

    pub fn empty() -> Self {
        Module {
            extl: Interface::empty(),
            intf: Interface::empty(),
            prvt: Interface::empty(),
            obs: Interface::empty(),
            ctrl: Interface::empty(),
            temp: Interface::empty(),
            atoms: Vec::new(),
        }
    }
}

impl<T: Theory> Module<T>
where
    T::Sort: Clone + Eq + Debug,
{
    /// Constructs a module **without performing any consistency or visibility checks**.
    ///
    /// This constructor provides **full control** to the caller and performs no inference
    /// or validation. It should be used only when all necessary checks or automation
    /// have already been handled externally.
    ///
    /// Unlike the other constructors, which automatically infer wire visibility and module
    /// properties, `new_unchecked` allows manually specifying all wire sets and atoms.
    /// This is useful for advanced scenarios or for optimising performance when redundant
    /// automation would otherwise occur.
    ///
    /// # Wire layout and visibility
    ///
    /// Wire visibility and organization within a module can be visualized as follows:
    ///
    /// ```text
    ///     *====================*
    ///     | extl | intf | prvt |
    ///     *--------------------*
    ///     |     obs     | prvt |
    ///     *--------------------*
    ///     | extl |    ctrl     |
    ///     *====================*
    /// ```
    /// # Parameters
    /// - `extl` are external wires, exposed to the environment (module inputs).
    /// - `intf` are interface wires, forming the module’s public outputs.
    /// - `prvt` are private wires, hidden from external access.
    /// - `obs` are observable wires, visible through the module interface.
    /// - `ctrl` are controlled wires, used for state management and internal coordination.
    /// - `atoms`: The list of atoms defining the module’s internal behaviour.
    ///
    /// # Returns
    /// The constructed module.
    ///
    /// # Safety
    /// This function performs **no validation or inference**. It is the caller’s
    /// responsibility to ensure that all wires, atoms, and interfaces are well-formed
    /// and consistent.
    ///
    /// # See Also
    /// - [`Atom::sequential`], [`Atom::combinatorial`] for creating individual atoms.
    /// - [`Module::partially_observable`], [`Module::observable`], [`Module::sequential_observable`],
    ///   [`Module::combinatorial`] for safe, automated module construction
    #[allow(clippy::too_many_arguments)]
    fn new_unchecked(
        extl: Interface<T::Sort, 2>,
        intf: Interface<T::Sort, 2>,
        prvt: Interface<T::Sort, 2>,
        obs: Interface<T::Sort, 2>,
        ctrl: Interface<T::Sort, 2>,
        temp: Interface<T::Sort>,
        atoms: Vec<Atom<T>>,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(obs.len(), extl.len() + intf.len());
            debug_assert_eq!(ctrl.len(), intf.len() + prvt.len());

            let mut ltc_to_dtype: HashMap<usize, &T::Sort> = HashMap::new();
            let mut nxt_to_ltc: HashMap<usize, usize> = HashMap::new();

            let mut extl_ltc: HashSet<usize> = HashSet::new();
            for [ltc, nxt] in &extl {
                debug_assert_eq!(ltc.dtype(), nxt.dtype());
                // check that indices are unique, and store them
                debug_assert!(ltc_to_dtype.insert(ltc.id(), ltc.dtype()).is_none());
                debug_assert!(nxt_to_ltc.insert(nxt.id(), ltc.id()).is_none());

                extl_ltc.insert(ltc.id());
            }

            let mut intf_ltc: HashSet<usize> = HashSet::new();
            for [ltc, nxt] in &intf {
                debug_assert_eq!(ltc.dtype(), nxt.dtype());
                // check that indices are unique, and store them
                debug_assert!(ltc_to_dtype.insert(ltc.id(), ltc.dtype()).is_none());
                debug_assert!(nxt_to_ltc.insert(nxt.id(), ltc.id()).is_none());

                intf_ltc.insert(ltc.id());
            }

            let mut prvt_ltc: HashSet<usize> = HashSet::new();
            for [ltc, nxt] in &prvt {
                debug_assert_eq!(ltc.dtype(), nxt.dtype());
                // check that indices are unique, and store them
                debug_assert!(ltc_to_dtype.insert(ltc.id(), ltc.dtype()).is_none());
                debug_assert!(nxt_to_ltc.insert(nxt.id(), ltc.id()).is_none());

                prvt_ltc.insert(ltc.id());
            }

            let mut obs_ltc: HashSet<usize> = HashSet::new();
            for [ltc, nxt] in &obs {
                debug_assert_eq!(ltc.dtype(), nxt.dtype());
                // check consistency with other wires
                debug_assert_eq!(ltc_to_dtype.get(&ltc.id()), Some(&ltc.dtype()));
                debug_assert_eq!(nxt_to_ltc.get(&nxt.id()), Some(&ltc.id()));
                // check that indices are unique
                debug_assert!(obs_ltc.insert(ltc.id()));

                debug_assert!(extl_ltc.contains(&ltc.id()) || intf_ltc.contains(&ltc.id()));
            }

            let mut ctrl_ltc: HashSet<usize> = HashSet::new();
            for [ltc, nxt] in &ctrl {
                debug_assert_eq!(ltc.dtype(), nxt.dtype());
                // check consistency with other wires
                debug_assert_eq!(ltc_to_dtype.get(&ltc.id()), Some(&ltc.dtype()));
                debug_assert_eq!(nxt_to_ltc.get(&nxt.id()), Some(&ltc.id()));
                // check that indices are unique
                debug_assert!(ctrl_ltc.insert(ltc.id()));

                debug_assert!(intf_ltc.contains(&ltc.id()) || prvt_ltc.contains(&ltc.id()));
            }

            let nxt_to_dtype_get = |w| nxt_to_ltc.get(&w).and_then(|z| ltc_to_dtype.get(z));

            // check atoms consistency
            let mut written: HashSet<usize> = HashSet::from_iter(extl.next().iter().map(Wire::id));
            for atom in atoms.iter() {
                for (ltc, dtype) in atom.read().wires().map(Into::into) {
                    // reads are latched, and dtype matches
                    debug_assert_eq!(Some(&dtype), ltc_to_dtype.get(&ltc));
                }
                for (nxt, dtype) in atom.wait().wires().map(Into::into) {
                    // awaits are next, and dtype matches
                    debug_assert_eq!(Some(&dtype), nxt_to_dtype_get(nxt));
                    // await order is consistent
                    debug_assert!(written.contains(&nxt));
                }
                for (nxt, dtype) in atom.ctrl().wires().map(Into::into) {
                    // controls are next, and dtype matches
                    debug_assert_eq!(Some(&dtype), nxt_to_dtype_get(nxt));
                    // controls are disjoint
                    debug_assert!(written.insert(nxt));
                }
            }

            // check that all module control wires are written/controlled by an atom
            for nxt in ctrl.next().iter().map(Wire::id) {
                debug_assert!(written.contains(&nxt));
            }

            // check that temporaries are decoupled from module wires and other atoms
            let mut module_temp: HashMap<usize, &T::Sort> = HashMap::new();
            for lc in atoms.iter().flat_map(Atom::temp) {
                debug_assert!(!ltc_to_dtype.contains_key(&lc.id()));
                debug_assert!(!nxt_to_ltc.contains_key(&lc.id()));
                debug_assert!(module_temp.insert(lc.id(), lc.dtype()).is_none());
            }
            debug_assert_eq!(module_temp.len(), temp.len());
            debug_assert!(temp.iter().all(|[wire]| {
                module_temp
                    .get(&wire.id())
                    .is_some_and(|&dtype| dtype == wire.dtype())
            }));
        }

        Module {
            extl,
            intf,
            prvt,
            obs,
            ctrl,
            temp,
            atoms,
        }
    }

    /// Constructs a **fully observable module** from a set of atoms.
    ///
    /// A fully observable module exposes all of its wires (`obs`) publicly, so that
    /// no internal state remains hidden. This is useful when the entire behaviour
    /// of the module should be visible through its interface.
    ///
    /// The module is composed of the provided atoms, and wire visibility is automatically
    /// inferred from the atoms. Unlike partially observable modules, there are no private wires,
    /// so the module’s interface is entirely transparent.
    ///
    /// # Parameters
    /// - `obs`: The pair of observable wires `[latched, next]` representing the module’s interface.
    /// - `atoms`: An iterable collection of atoms defining the module’s internal behaviour.
    ///
    /// # Returns
    /// A `Result` containing the constructed fully observable module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`partially_observable`], for modules with private state.
    /// - [`Atom::sequential`], [`Atom::combinatorial`] for creating individual atoms.
    /// - [`new_unchecked`], for manual module creation.
    pub fn observable<D, O, A>(obs: O, atoms: A) -> Result<Self, Error>
    where
        D: Into<[Wire<T::Sort>; 2]>,
        O: IntoIterator<Item = D>,
        A: IntoIterator<Item = Atom<T>> + Sized,
    {
        Self::partially_observable(obs, std::iter::empty::<D>(), atoms)
    }

    /// Constructs a **partially observable module** from a sequence of atoms.
    ///
    /// A partially observable module exposes only a subset of its wires (`obs`) while
    /// keeping others private (`prvt`). This allows encapsulation of internal state
    /// or logic that should not be visible externally.
    ///
    /// The module is composed of the provided atoms, and the visibility of each wire
    /// is automatically inferred from the atoms. Unlike fully observable modules,
    /// some internal wires remain hidden, giving the user control over the module’s interface.
    ///
    /// # Parameters
    /// - `obs`: The pair of observable wires `[latched, next]` representing the module’s interface.
    /// - `prvt`: The pair of private wires that remain hidden from external access.
    /// - `atoms`: An iterable collection of atoms defining the module’s internal behaviour.
    ///
    /// # Returns
    /// A `Result` containing the constructed partially observable module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`observable`], for constructing modules where all wires are visible.
    /// - [`Atom::sequential`], [`Atom::combinatorial`] for creating individual atoms.
    /// - [`new_unchecked`], for manual module creation.
    pub fn partially_observable<R, U, O, P, A>(obs: O, prvt: P, atoms: A) -> Result<Self, Error>
    where
        R: Into<[Wire<T::Sort>; 2]>,
        U: Into<[Wire<T::Sort>; 2]>,
        O: IntoIterator<Item = R>,
        P: IntoIterator<Item = U>,
        A: IntoIterator<Item = Atom<T>> + Sized,
    {
        let mut ltc_set: HashSet<usize> = HashSet::new();
        let mut nxt_to_ltc: HashMap<usize, usize> = HashMap::new();

        let obs = Interface::try_from_iter(obs)?;
        let prvt = Interface::try_from_iter(prvt)?;

        for [ltc, nxt] in obs.iter().chain(prvt.iter()) {
            debug_assert_eq!(ltc.dtype(), nxt.dtype());
            if !ltc_set.insert(ltc.id()) {
                return Err(format!("Latched wire {} is duplicated", ltc.id()));
            }
            if nxt_to_ltc.insert(nxt.id(), ltc.id()).is_some() {
                return Err(format!("Next wire {} is duplicated", ltc.id()));
            }
        }

        // Check atoms consistency and infer control wires
        let mut ctrl_nxt: HashSet<usize> = HashSet::new();
        let mut temp: BTreeMap<usize, Wire<T::Sort>> = BTreeMap::new();
        let atoms_iter = atoms.into_iter();
        let mut past_atoms: Vec<Atom<T>> = Vec::with_capacity(atoms_iter.size_hint().0);
        for (n, atom) in atoms_iter.enumerate() {
            for ltc in atom.read().wires().map(Wire::id) {
                if !ltc_set.contains(&ltc) {
                    return Err(format!("Invalid read wire {} in atom {}", ltc, n));
                }
            }
            for nxt in atom.wait().wires().map(Wire::id) {
                let expected_dtype = nxt_to_ltc.get(&nxt);
                if expected_dtype.is_none_or(|i| !ltc_set.contains(i)) {
                    return Err(format!("invalid await wire {} in atom {}", nxt, n));
                }
            }
            for nxt in atom.ctrl().wires().map(Wire::id) {
                let expected_dtype = nxt_to_ltc.get(&nxt);
                if expected_dtype.is_none_or(|i| !ltc_set.contains(i)) {
                    return Err(format!("invalid control wire {} in atom {}", nxt, n));
                }
                if !ctrl_nxt.insert(nxt) {
                    return Err(format!(
                        "shared or duplicated control wire {} in atom {}",
                        nxt, n
                    ));
                }
            }

            for lc in atom.temp() {
                debug_assert!(!ctrl_nxt.contains(&lc.id()));
                if ltc_set.contains(&lc.id()) || nxt_to_ltc.contains_key(&lc.id()) {
                    return Err(format!("temp wire {} is also a module wire", lc.id()));
                }
                if temp.insert(lc.id(), lc.clone()).is_some() {
                    return Err(format!("temp wire {} coupled with other atom", lc.id()));
                }
            }

            for past_atom in &past_atoms {
                if past_atom.awaits(&atom) {
                    return Err(format!(
                        "Atom {} is awaited by some previous atom, inconsistent awaiting order",
                        n
                    ));
                }
            }
            past_atoms.push(atom);
        }

        // Check that private wires are controlled
        for nxt in prvt.next().iter().map(Wire::id) {
            if !ctrl_nxt.contains(&nxt) {
                return Err(format!("private wire {} is not controlled", nxt));
            }
        }

        // Build intf and extl wires based on inferred control set
        let mut intf: Vec<[Wire<T::Sort>; 2]> = Vec::with_capacity(ctrl_nxt.len() - prvt.len());
        let mut extl: Vec<[Wire<T::Sort>; 2]> = Vec::with_capacity(obs.len() - intf.len());
        let mut ctrl: Vec<[Wire<T::Sort>; 2]> = Vec::with_capacity(ctrl_nxt.len());

        for [ltc, nxt] in obs.iter() {
            if ctrl_nxt.contains(&nxt.id()) {
                intf.push([ltc.clone(), nxt.clone()]);
                ctrl.push([ltc.clone(), nxt.clone()]);
            } else {
                extl.push([ltc.clone(), nxt.clone()]);
            }
        }

        for [ltc, nxt] in prvt.iter() {
            ctrl.push([ltc.clone(), nxt.clone()]);
        }

        let extl = Interface::from_iter_unchecked(extl);
        let ctrl = Interface::from_iter_unchecked(ctrl);
        let intf = Interface::from_iter_unchecked(intf);
        let temp = Interface::from_wires_unchecked(temp.into_values());

        Ok(Self::new_unchecked(
            extl, intf, prvt, obs, ctrl, temp, past_atoms,
        ))
    }

    /// Constructs a **sequential module** from an initialisation and update sequences of terms.
    ///
    /// A sequential module represents **time-dependent behaviour**, with state evolving
    /// across discrete steps. It is composed of a single [`Atom::sequential`] atom,
    /// and is **fully observable by default**.
    ///
    /// # Parameters
    /// - `obs`: The sequence of `[latched, next]`-wire pairs representing the module’s observables.
    /// - `prvt`: The sequence of `[latched, next]`-wire pairs representing the module’s hidden state.
    /// - `init`: The set of terms defining the module’s initial state.
    /// - `update`: The set of terms defining the module’s state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed sequential module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::combinatorial`], for constructing stateless, time-independent modules.
    /// - [`Atom::sequential`], for creating individual sequential atoms.
    pub fn sequential_observable<R, O, V, U>(obs: O, init: V, update: U) -> Result<Self, Error>
    where
        R: Into<[Wire<T::Sort>; 2]>,
        O: IntoIterator<Item = R>,
        V: IntoIterator<Item = Term<T>>,
        U: IntoIterator<Item = Term<T>>,
    {
        Self::sequential(obs, std::iter::empty::<R>(), init, update)
    }

    pub fn sequential<TO, TP, O, P, V, U>(
        obs: O,
        prvt: P,
        init: V,
        update: U,
    ) -> Result<Self, Error>
    where
        TO: Into<[Wire<T::Sort>; 2]>,
        TP: Into<[Wire<T::Sort>; 2]>,
        O: IntoIterator<Item = TO>,
        P: IntoIterator<Item = TP>,
        V: IntoIterator<Item = Term<T>>,
        U: IntoIterator<Item = Term<T>>,
    {
        let obs = Interface::try_from_iter(obs)?;
        let prvt = Interface::try_from_iter(prvt)?;
        let latched = obs.latched().iter().chain(prvt.latched().iter());
        let next = obs.next().iter().chain(prvt.next().iter());
        let atom = Atom::sequential(latched, next, init, update)?;
        Self::partially_observable(obs, prvt, [atom])
    }

    /// Constructs the *parallel composition* of several `Module` instances.
    ///
    /// This function takes an iterator of modules and returns a new module that
    /// represents all of them composed in parallel, coupling all shared observable
    /// wires.
    ///
    /// # Semantics
    ///
    /// Observable wires with identical id across modules are *coupled* in the composed
    /// module. Coupling means that these wires represent the same value in the resulting system.
    ///
    /// # Error Conditions
    ///
    /// - A module attempts to couple a *private* or *temporary* wire with another module
    /// - A coupled wire is *controlled by more than one module*
    /// - Await dependency is cyclic
    ///
    /// # Returns
    ///
    /// - `Ok(Module<D, I>)` containing the composed module.
    /// - `Err(Error)` describing the reason composition failed.
    ///
    pub fn parallel<M>(modules: M) -> Result<Self, Error>
    where
        M: IntoIterator<Item = Self>,
    {
        let mut latched: HashSet<usize> = HashSet::new();
        let mut next: HashSet<usize> = HashSet::new();
        let mut restricted: HashSet<usize> = HashSet::new();

        let mut extl: HashMap<usize, Wire<T::Sort>> = HashMap::new();
        let mut intf: HashMap<usize, Wire<T::Sort>> = HashMap::new();

        let mut extl_stack: Vec<[Wire<T::Sort>; 2]> = Vec::new();
        let mut intf_stack: Vec<[Wire<T::Sort>; 2]> = Vec::new();
        let mut prvt_stack: Vec<[Wire<T::Sort>; 2]> = Vec::new();
        let mut obs_stack: Vec<[Wire<T::Sort>; 2]> = Vec::new();
        let mut ctrl_stack: Vec<[Wire<T::Sort>; 2]> = Vec::new();
        let mut temp_stack: Vec<[Wire<T::Sort>; 1]> = Vec::new();
        let mut atoms_stack: Vec<Atom<T>> = Vec::new();

        let mut await_graph: Vec<Vec<usize>> = Vec::new();

        for module in modules {
            //============================================================
            // Ensure decoupling and restrict visibility
            //============================================================

            // Check that observables are either uncoupled or coupled in right direction
            obs_stack.reserve(module.obs.len());
            for [ltc, nxt] in module.obs {
                debug_assert_eq!(ltc.dtype(), nxt.dtype());
                if latched.contains(&nxt.id()) || next.contains(&ltc.id()) {
                    return Err(format!(
                        "invalid coupling (direction): either {} is latched or {} is next",
                        nxt.id(),
                        ltc.id()
                    ));
                }
                if restricted.contains(&ltc.id()) || restricted.contains(&nxt.id()) {
                    return Err(format!(
                        "invalid coupling for wire pair ({}, {}) (restricted)",
                        ltc.id(),
                        nxt.id()
                    ));
                }
                // stack observables that are not already present (avoid duplication)
                if latched.insert(ltc.id()) {
                    debug_assert!(!next.contains(&nxt.id()));
                    next.insert(nxt.id());
                    obs_stack.push([ltc, nxt]);
                }
            }

            // Check that privates are uncoupled and restrict them
            prvt_stack.reserve(module.prvt.len());
            for [ltc, nxt] in module.prvt {
                debug_assert_eq!(ltc.dtype(), nxt.dtype());
                if !latched.insert(ltc.id()) || !next.insert(nxt.id()) {
                    return Err(format!(
                        "invalid coupling for wire pair ({}, {}) (private)",
                        ltc.id(),
                        nxt.id()
                    ));
                }
                if latched.contains(&nxt.id()) || next.contains(&ltc.id()) {
                    return Err(format!(
                        "invalid coupling for wire pair ({}, {}) (direction)",
                        ltc.id(),
                        nxt.id()
                    ));
                }
                debug_assert!(!restricted.contains(&ltc.id()) && !restricted.contains(&nxt.id()));
                restricted.insert(ltc.id());
                restricted.insert(nxt.id());

                prvt_stack.push([ltc, nxt]);
            }

            // Check that temporaries are uncoupled and restrict them
            temp_stack.reserve(module.temp.len());
            for [tmp] in module.temp {
                if latched.contains(&tmp.id()) || next.contains(&tmp.id()) {
                    return Err(format!("Temp wire {} is latched or next", tmp.id()));
                }
                if !restricted.insert(tmp.id()) {
                    return Err(format!(
                        "invalid coupling of temp wire {} (is in restricted)",
                        tmp.id()
                    ));
                }

                temp_stack.push([tmp]);
            }

            //============================================================
            // Couple external and interface variables
            //============================================================
            extl_stack.reserve(module.extl.len());
            for [ltc, nxt] in module.extl {
                debug_assert_eq!(ltc.dtype(), nxt.dtype());
                if restricted.contains(&ltc.id()) || restricted.contains(&nxt.id()) {
                    return Err(format!(
                        "Wire pair ({}, {}) is restricted",
                        ltc.id(),
                        nxt.id()
                    ));
                }

                // check whether the wire is coupled (controlled by other atom), or
                // consider it as external otherwise
                if let Some(coupled) = intf.get(&ltc.id()) {
                    if coupled.id() != nxt.id() {
                        return Err(format!(
                            "wire id mismatch, expected that wire {} will be the same as {}",
                            coupled.id(),
                            nxt.id()
                        ));
                    } else if coupled.dtype() != nxt.dtype() {
                        return Err(format!("Wire {} has a wrong dtype", coupled.id()));
                    }
                } else {
                    extl.insert(ltc.id(), nxt.clone());
                    extl_stack.push([ltc, nxt]);
                }
            }

            intf_stack.reserve(module.intf.len());
            for [ltc, nxt] in module.intf {
                debug_assert_eq!(ltc.dtype(), nxt.dtype());
                if restricted.contains(&ltc.id()) || restricted.contains(&nxt.id()) {
                    return Err(format!(
                        "Wire pair ({}, {}) is restricted",
                        ltc.id(),
                        nxt.id()
                    ));
                }

                // check whether the wire is coupled (external of other atom), and
                // consider it as interface wire then
                if let Some(coupled) = extl.remove(&ltc.id()) {
                    if coupled.id() != nxt.id() {
                        return Err(format!(
                            "wire id mismatch, expected that wire {} will be the same as {}",
                            coupled.id(),
                            nxt.id()
                        ));
                    } else if coupled.dtype() != nxt.dtype() {
                        return Err(format!("Wire {} has a wrong dtype", coupled.id()));
                    }
                }

                if intf.insert(ltc.id(), nxt.clone()).is_some() {
                    return Err(format!(
                        "invalid coupling for wire pair ({}, {}), shared control",
                        ltc.id(),
                        nxt.id()
                    ));
                }

                intf_stack.push([ltc, nxt]);
            }

            ctrl_stack.extend(module.ctrl);

            //============================================================
            // Populate await graph
            //============================================================
            for this_atom in module.atoms {
                let this_idx = atoms_stack.len();
                let mut this_adj = Vec::new();
                for (other_idx, other_atom) in atoms_stack.iter().enumerate() {
                    if this_atom.awaits(other_atom) {
                        await_graph[other_idx].push(this_idx);
                    }
                    if other_atom.awaits(&this_atom) {
                        this_adj.push(other_idx);
                    }
                }
                atoms_stack.push(this_atom);
                await_graph.push(this_adj);
            }
        }

        //============================================================
        // Reorder atoms and remove coupled wires from the externals
        //============================================================

        let await_order = topological_order(&await_graph).ok_or("invalid await dependency")?;
        debug_assert_eq!(await_order.len(), await_graph.len());

        let mut atoms: Vec<Atom<T>> = Vec::with_capacity(await_graph.len());
        for idx in await_order {
            atoms.push(std::mem::take(&mut atoms_stack[idx]));
        }

        let extl_stack = extl_stack
            .into_iter()
            .filter(|wire| extl.contains_key(&wire[0].id()));

        //============================================================
        // Collect and construct
        //============================================================

        let extl = Interface::from_iter_unchecked(extl_stack);
        let intf = Interface::from_iter_unchecked(intf_stack);
        let prvt = Interface::from_iter_unchecked(prvt_stack);
        let obs = Interface::from_iter_unchecked(obs_stack);
        let ctrl = Interface::from_iter_unchecked(ctrl_stack);
        let temp = Interface::from_iter_unchecked(temp_stack);

        Ok(Module::new_unchecked(
            extl, intf, prvt, obs, ctrl, temp, atoms,
        ))
    }
}

impl<T: Theory> Module<T>
where
    T: Clone,
    T::Sort: Eq + Clone + Debug,
{
    /// Constructs a **purely combinatorial module** from an assignment sequence of terms.
    ///
    /// A combinatorial module represents a **stateless, time-independent** relationship
    /// between observable wires. It is composed of a single [`Atom::combinatorial`] atom,
    /// and is **fully observable by default**.
    ///
    /// # Parameters
    /// - `obs`: The pair of observable wires `[latched, next]` representing the module’s interface.
    /// - `assign`: The set of combinatorial assignment terms defining how the output is
    ///   computed from the input.
    ///
    /// # Returns
    /// A `Result` containing the constructed combinatorial module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::sequential_observable`], for constructing stateful, sequential modules.
    /// - [`Atom::combinatorial`], for creating individual combinatorial atoms.
    pub fn combinatorial<R, O, V>(obs: O, assign: V) -> Result<Self, Error>
    where
        R: Into<[Wire<T::Sort>; 2]>,
        O: IntoIterator<Item = R>,
        V: IntoIterator<Item = Term<T>>,
    {
        let obs = Interface::from_iter(obs);
        let atom = Atom::combinatorial(obs.next(), assign)?;
        Self::observable(obs, [atom])
    }
}

impl<T: Theory> Module<T>
where
    T: fmt::Display,
    T::Sort: fmt::Display,
{
    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, pad: &str) -> fmt::Result {
        const BOLD: &str = "\x1b[1m";
        const RESET: &str = "\x1b[0m";
        const INDENT: &str = "  ";
        const INDENT2: &str = "    ";

        writeln!(f, "{pad}{BOLD}module{RESET}")?;
        if !self.extl.is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}external{RESET}")?;
        }
        for [(ltc, _), (nxt, dtype)] in self.extl.iter().map(|w| w.map(Into::into)) {
            writeln!(f, "{pad}{INDENT2}w{ltc}, w{nxt} : {dtype}")?;
        }
        if !self.intf.is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}interface{RESET}")?;
        }
        for [(ltc, _), (nxt, dtype)] in self.intf.iter().map(|w| w.map(Into::into)) {
            writeln!(f, "{pad}{INDENT2}w{ltc}, w{nxt} : {dtype}")?;
        }
        if !self.prvt.is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}private{RESET}")?;
        }
        for [(ltc, _), (nxt, dtype)] in self.prvt.iter().map(|w| w.map(Into::into)) {
            writeln!(f, "{pad}{INDENT2}w{ltc}, w{nxt} : {dtype}")?;
        }
        for atom in &self.atoms {
            atom.fmt_indent(f, &format!("{pad}{INDENT}"))?;
        }
        Ok(())
    }
}

impl<T: Theory> fmt::Display for Module<T>
where
    T: fmt::Display,
    T::Sort: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, "")
    }
}
