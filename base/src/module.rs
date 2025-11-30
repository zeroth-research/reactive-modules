use crate::atom::Atom;
use crate::term::Term;
use crate::wire::{Interface, Wire};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::fmt::Debug;

/// This data structure corresponds to the module of reactive modules.
#[derive(Debug)]
pub struct Module<D, I> {
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
    extl: Interface<D, 2>,
    intf: Interface<D, 2>,
    prvt: Interface<D, 2>,
    obs: Interface<D, 2>,
    ctrl: Interface<D, 2>,

    /// The atoms of this module.
    /// The atoms must be stored in a *consistent* linear order
    /// as defined in the reactive modules paper.
    atoms: Vec<Atom<D, I>>,
}

impl<D, I> Module<D, I> {
    pub fn atoms(&self) -> &[Atom<D, I>] {
        &self.atoms
    }

    pub fn extl(&self) -> &Interface<D, 2> {
        &self.extl
    }

    pub fn intf(&self) -> &Interface<D, 2> {
        &self.intf
    }

    pub fn prvt(&self) -> &Interface<D, 2> {
        &self.prvt
    }

    pub fn ctrl(&self) -> &Interface<D, 2> {
        &self.ctrl
    }

    pub fn obs(&self) -> &Interface<D, 2> {
        &self.obs
    }

    pub fn is_closed(&self) -> bool {
        self.extl.is_empty()
    }

    pub fn is_open(&self) -> bool {
        !self.extl.is_empty()
    }
}

impl<D: Clone + Eq + Debug, I> Module<D, I> {
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
    /// - [`Module::partially_observable`], [`Module::observable`], [`Module::sequential`],
    ///   [`Module::combinatorial`] for safe, automated module construction
    pub fn new_unchecked(
        extl: Interface<D, 2>,
        intf: Interface<D, 2>,
        prvt: Interface<D, 2>,
        obs: Interface<D, 2>,
        ctrl: Interface<D, 2>,
        atoms: Vec<Atom<D, I>>,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(obs.len(), extl.len() + intf.len());
            debug_assert_eq!(ctrl.len(), intf.len() + prvt.len());

            let mut ltc_to_dtype: HashMap<usize, &D> = HashMap::new();
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
            let mut written: HashSet<usize> = HashSet::from_iter(extl[1].iter().map(Wire::id));
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
            for nxt in ctrl[1].iter().map(Wire::id) {
                debug_assert!(written.contains(&nxt));
            }

            // check that temporaries are decoupled from module wires and other atoms
            let mut module_temp = HashSet::new();
            for lc in atoms.iter().map(Atom::temp).flat_map(Interface::ids) {
                debug_assert!(!ltc_to_dtype.contains_key(&lc));
                debug_assert!(!nxt_to_ltc.contains_key(&lc));
                debug_assert!(module_temp.insert(lc));
            }
        }

        Module {
            extl,
            intf,
            prvt,
            obs,
            ctrl,
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
    pub fn observable<T, O, A>(obs: O, atoms: A) -> Result<Self, &'static str>
    where
        T: Into<[Wire<D>; 2]>,
        O: IntoIterator<Item = T>,
        A: IntoIterator<Item = Atom<D, I>> + Sized,
    {
        let prvt: std::iter::Empty<[Wire<D>; 2]> = std::iter::empty();
        Self::partially_observable(obs, prvt, atoms)
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
    pub fn partially_observable<T, U, O, P, A>(
        obs: O,
        prvt: P,
        atoms: A,
    ) -> Result<Self, &'static str>
    where
        T: Into<[Wire<D>; 2]>,
        U: Into<[Wire<D>; 2]>,
        O: IntoIterator<Item = T>,
        P: IntoIterator<Item = U>,
        A: IntoIterator<Item = Atom<D, I>> + Sized,
    {
        let mut ltc_to_dtype: HashMap<usize, &D> = HashMap::new();
        let mut nxt_to_ltc: HashMap<usize, usize> = HashMap::new();

        let obs = Interface::try_from_iter(obs)?;
        let prvt = Interface::try_from_iter(prvt)?;

        for [ltc, nxt] in obs.iter().chain(prvt.iter()) {
            debug_assert_eq!(ltc.dtype(), nxt.dtype());
            // if ltc.dtype() != nxt.dtype() {
            //     return Err("dtype mismatch in latched and next wires");
            // }
            if ltc_to_dtype.insert(ltc.id(), ltc.dtype()).is_some() {
                return Err("duplicate latched wire");
            }
            if nxt_to_ltc.insert(nxt.id(), ltc.id()).is_some() {
                return Err("duplicate next wire");
            }
        }

        // Check atoms consistency and infer control wires
        let mut ctrl_nxt: HashSet<usize> = HashSet::new();
        let mut temp: BTreeMap<usize, D> = BTreeMap::new();
        let atoms_iter = atoms.into_iter();
        let mut past_atoms: Vec<Atom<D, I>> = Vec::with_capacity(atoms_iter.size_hint().0);
        for atom in atoms_iter {
            for (ltc, dtype) in atom.read().wires().map(Into::into) {
                if ltc_to_dtype.get(&ltc) != Some(&dtype) {
                    return Err("invalid read wire or dtype mismatch");
                }
            }
            for (nxt, dtype) in atom.wait().wires().map(Into::into) {
                let expected_dtype = nxt_to_ltc.get(&nxt);
                if expected_dtype.is_none_or(|i| ltc_to_dtype.get(i) != Some(&dtype)) {
                    return Err("invalid await wire or dtype mismatch");
                }
            }
            for (nxt, dtype) in atom.ctrl().wires().map(Into::into) {
                let expected_dtype = nxt_to_ltc.get(&nxt);
                if expected_dtype.is_none_or(|i| ltc_to_dtype.get(i) != Some(&dtype)) {
                    return Err("invalid control wire or dtype mismatch");
                }
                if !ctrl_nxt.insert(nxt) {
                    return Err("shared or duplicate atom control wire");
                }
            }

            for (lc, dtype) in atom.temp().wires().map(Into::into) {
                debug_assert!(!ctrl_nxt.contains(&lc));
                if ltc_to_dtype.contains_key(&lc) || nxt_to_ltc.contains_key(&lc) {
                    return Err("temp wires coupled with module wires");
                }
                if temp.insert(lc, dtype.clone()).is_some() {
                    return Err("temp wires could with other atom");
                }
            }

            for past_atom in &past_atoms {
                if past_atom.awaits(&atom) {
                    return Err("inconsistent awaiting order");
                }
            }
            past_atoms.push(atom);
        }

        // Check that private wires are controlled
        for nxt in prvt[1].iter().map(Wire::id) {
            if !ctrl_nxt.contains(&nxt) {
                return Err("private wire not controlled");
            }
        }

        // Build intf and extl wires based on inferred control set
        let mut extl: Vec<[Wire<D>; 2]> = Vec::with_capacity(obs[0].len() - ctrl_nxt.len());
        let mut intf: Vec<[Wire<D>; 2]> = Vec::with_capacity(ctrl_nxt.len() - prvt[0].len());
        let mut ctrl: Vec<[Wire<D>; 2]> = Vec::with_capacity(ctrl_nxt.len());

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

        let extl = Interface::<D, 2>::from_iter_unchecked(extl);
        let ctrl = Interface::<D, 2>::from_iter_unchecked(ctrl);
        let intf = Interface::<D, 2>::from_iter_unchecked(intf);

        Ok(Self::new_unchecked(extl, intf, prvt, obs, ctrl, past_atoms))
    }

    /// Constructs a **sequential module** from an initialisation and update sequences of terms.
    ///
    /// A sequential module represents **time-dependent behaviour**, with state evolving
    /// across discrete steps. It is composed of a single [`Atom::sequential`] atom,
    /// and is **fully observable by default**.
    ///
    /// # Parameters
    /// - `obs`: The pair of observable wires `[latched, next]` representing the module’s interface.
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
    pub fn sequential<T, O, V, U>(obs: O, init: V, update: U) -> Result<Self, &'static str>
    where
        T: Into<[Wire<D>; 2]>,
        O: IntoIterator<Item = T>,
        V: IntoIterator<Item = Term<D, I>>,
        U: IntoIterator<Item = Term<D, I>>,
    {
        let obs = Interface::from_iter(obs);
        let atom = Atom::sequential(&obs[0], &obs[1], init, update)?;
        Self::observable(obs, [atom])
    }
}

impl<D: Eq + Clone + Debug, I: Clone> Module<D, I> {
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
    /// - [`Module::sequential`], for constructing stateful, sequential modules.
    /// - [`Atom::combinatorial`], for creating individual combinatorial atoms.
    pub fn combinatorial<T, O, V>(obs: O, assign: V) -> Result<Self, &'static str>
    where
        T: Into<[Wire<D>; 2]>,
        O: IntoIterator<Item = T>,
        V: IntoIterator<Item = Term<D, I>>,
    {
        let obs = Interface::from_iter(obs);
        let atom = Atom::combinatorial(&obs[1], assign)?;
        Self::observable(obs, [atom])
    }
}

impl<D: fmt::Display, I: fmt::Display> Module<D, I> {
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

impl<D: fmt::Display, I: fmt::Display> fmt::Display for Module<D, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, "")
    }
}
