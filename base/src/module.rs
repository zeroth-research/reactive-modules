use crate::atom::Atom;
use crate::topological_order;
use crate::var::{Interface, Var};
use crate::wire::Wire;
use std::borrow::Cow;
use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::fmt::Debug;
use theory::{Combinatorial, Differential, Sequential, Theory};

//============================================================
// Module
//============================================================

/// This data structure corresponds to the module of reactive modules.
#[derive(Debug, Clone)]
pub struct Module<I, J, F, S = <I as Theory>::Sort>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
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
    extl: Interface<S>,
    intf: Interface<S>,
    prvt: Interface<S>,
    obs: Interface<S>,
    ctrl: Interface<S>,

    /// The atoms of this module.
    /// The atoms must be stored in a *consistent* linear order
    /// according to their await relation.
    atoms: Vec<Atom<I, J, F, S>>,

    /// cache of all wires local to blocks
    local: Vec<Wire<S>>,
    /// cache of all local and global wires
    wires: Vec<Wire<S>>,
}

impl<I, J, F, S> Module<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    pub fn atoms(&self) -> &[Atom<I, J, F, S>] {
        &self.atoms
    }

    pub fn extl(&self) -> &Interface<S> {
        &self.extl
    }

    pub fn intf(&self) -> &Interface<S> {
        &self.intf
    }

    pub fn prvt(&self) -> &Interface<S> {
        &self.prvt
    }

    pub fn ctrl(&self) -> &Interface<S> {
        &self.ctrl
    }

    pub fn obs(&self) -> &Interface<S> {
        &self.obs
    }

    pub fn is_closed(&self) -> bool {
        self.extl.is_empty()
    }

    pub fn is_open(&self) -> bool {
        !self.extl.is_empty()
    }

    pub(crate) fn local(&self) -> &[Wire<S>] {
        self.local.as_slice()
    }

    pub(crate) fn wires(&self) -> &[Wire<S>] {
        self.wires.as_slice()
    }

    pub fn empty() -> Self {
        Module {
            extl: Interface::empty(),
            intf: Interface::empty(),
            prvt: Interface::empty(),
            obs: Interface::empty(),
            ctrl: Interface::empty(),
            atoms: Vec::new(),
            wires: Vec::new(),
            local: Vec::new(),
        }
    }
}

//============================================================
// Private routines
//============================================================

impl<I, J, F, S> Module<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Debug,
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
    /// - [`Module::partially_observable`], [`Module::observable`], [`Module::partially_observable_sequential`],
    ///   [`Module::combinatorial`] for safe, automated module construction
    #[allow(clippy::too_many_arguments)]
    fn new_unchecked(
        extl: Interface<S>,
        intf: Interface<S>,
        prvt: Interface<S>,
        obs: Interface<S>,
        ctrl: Interface<S>,
        atoms: Vec<Atom<I, J, F, S>>,
        wires: Vec<Wire<S>>,
        local: Vec<Wire<S>>,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(obs.len(), extl.len() + intf.len());
            debug_assert_eq!(ctrl.len(), intf.len() + prvt.len());

            let mut module_vars: HashSet<&Var<S>> = HashSet::new();

            // extl & intf & prvt = empty
            debug_assert!(extl.iter().all(|v| module_vars.insert(v)));
            debug_assert!(intf.iter().all(|v| module_vars.insert(v)));
            debug_assert!(prvt.iter().all(|v| module_vars.insert(v)));

            // obs U ctrl <= extl U intf U prvt
            debug_assert!(obs.iter().all(|v| module_vars.contains(v)));
            debug_assert!(ctrl.iter().all(|v| module_vars.contains(v)));
            // obs <= extl U intf and ctrl <= intf U prvt
            debug_assert!(obs.iter().all(|v| extl.contains(v) || intf.contains(v)));
            debug_assert!(ctrl.iter().all(|v| intf.contains(v) || prvt.contains(v)));

            // check atoms consistency
            let mut written: HashSet<&Var<S>> = HashSet::new();
            written.extend(extl.iter());
            for atom in atoms.iter() {
                for var in atom.read().iter() {
                    debug_assert!(module_vars.contains(var));
                }
                for var in atom.wait().iter() {
                    debug_assert!(module_vars.contains(var));
                    debug_assert!(written.contains(var));
                }
                for var in atom.ctrl().iter() {
                    debug_assert!(module_vars.contains(var));
                    debug_assert!(written.insert(var));
                }

                debug_assert!(atom.wires().iter().all(|w| wires.binary_search(w).is_ok()));
                debug_assert!(atom.local().iter().all(|w| local.binary_search(w).is_ok()));
            }

            // check that all module control vars are written/controlled by an atom
            for var in ctrl.iter() {
                debug_assert!(written.contains(var));
            }

            // check vars consistency
            let mut atom_vars = HashSet::new();
            atom_vars.extend(atoms.iter().flat_map(Atom::read));
            atom_vars.extend(atoms.iter().flat_map(Atom::ctrl));
            atom_vars.extend(atoms.iter().flat_map(Atom::wait));
            debug_assert_eq!(atom_vars, module_vars);

            // check wires consistency
            debug_assert!(wires.is_sorted());
            debug_assert!(local.is_sorted());
            debug_assert!(wires.windows(2).all(|w| w[0] < w[1]));
            debug_assert!(local.windows(2).all(|w| w[0] < w[1]));

            let mut block_wires = HashSet::new();
            block_wires.extend(atoms.iter().flat_map(Atom::init).flat_map(|b| b.read()));
            block_wires.extend(atoms.iter().flat_map(Atom::init).flat_map(|b| b.write()));
            block_wires.extend(atoms.iter().flat_map(Atom::delay).flat_map(|b| b.read()));
            block_wires.extend(atoms.iter().flat_map(Atom::delay).flat_map(|b| b.write()));
            block_wires.extend(atoms.iter().flat_map(Atom::update).flat_map(|b| b.read()));
            block_wires.extend(atoms.iter().flat_map(Atom::update).flat_map(|b| b.write()));

            let atom_wires: HashSet<_> = atom_vars.into_iter().flat_map(Var::wires).collect();
            let local_wires: HashSet<_> = block_wires.difference(&atom_wires).cloned().collect();
            let all_wires: HashSet<_> = local_wires.union(&atom_wires).cloned().collect();

            // wires == all_wires and local == local_wires
            debug_assert!(wires.iter().all(|w| all_wires.contains(w)));
            debug_assert!(all_wires.iter().all(|w| wires.binary_search(w).is_ok()));
            debug_assert!(local.iter().all(|w| local_wires.contains(w)));
            debug_assert!(local_wires.iter().all(|w| local.binary_search(w).is_ok()));
        }

        Module {
            extl,
            intf,
            prvt,
            obs,
            ctrl,
            atoms,
            wires,
            local,
        }
    }
}

//============================================================
// Public constructors
//============================================================

impl<I, J, F, S> Module<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Clone + Debug,
{
    /// Constructs a **fully observable module** from a set of atoms.
    ///
    /// A fully observable module exposes all of its variables publicly, so
    /// that no internal state remains hidden. The variables and their roles
    /// (external or interface) are inferred entirely from the atoms: the
    /// controlled variables become the interface, the remaining read or
    /// awaited variables are external. Equivalent to
    /// [`Module::partially_observable`] with nothing hidden.
    ///
    /// # Parameters
    /// - `atoms`: An iterable collection of atoms defining the module's behaviour.
    ///
    /// # Returns
    /// A `Result` containing the constructed fully observable module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::partially_observable`], for modules with private state.
    /// - [`Atom::sequential`], [`Atom::combinatorial`] for creating individual atoms.
    pub fn observable<A>(atoms: A) -> Result<Self, String>
    where
        A: IntoIterator<Item = Atom<I, J, F, S>> + Sized,
    {
        Self::partially_observable(atoms, |_| false)
    }

    /// Constructs a **partially observable module** from a sequence of atoms,
    /// hiding the given variables.
    ///
    /// The variables and their roles are inferred from the atoms — the
    /// controlled variables form the interface, the remaining read or awaited
    /// variables are external — and the `hide` variables are then moved
    /// from the interface into the private state. Only controlled variables
    /// can be hidden: privates must be controlled by the module.
    ///
    /// # Parameters
    /// - `atoms`: An iterable collection of atoms defining the module's behaviour.
    /// - `prvt`: The variables to hide from the environment; hiding nothing
    ///   yields a fully observable module.
    ///
    /// # Returns
    /// A `Result` containing the constructed partially observable module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::observable`], the fully observable special case.
    /// - [`Atom::sequential`], [`Atom::combinatorial`] for creating individual atoms.
    pub fn partially_observable<A, H>(atoms: A, hide: H) -> Result<Self, String>
    where
        A: IntoIterator<Item = Atom<I, J, F, S>> + Sized,
        H: Fn(&Var<S>) -> bool,
    {
        let mut wires: BTreeSet<Wire<S>> = BTreeSet::new();
        let mut local: BTreeSet<Wire<S>> = BTreeSet::new();

        let mut extl: BTreeSet<Var<S>> = BTreeSet::new();
        let mut intf: BTreeSet<Var<S>> = BTreeSet::new();
        let atoms = atoms.into_iter();
        let mut past_atoms: Vec<Atom<I, J, F, S>> = Vec::with_capacity(atoms.size_hint().0);
        for (n, atom) in atoms.enumerate() {
            extl.extend(atom.read().iter().cloned());
            extl.extend(atom.wait().iter().cloned());

            for var in atom.ctrl().iter() {
                extl.remove(var);
                if !intf.insert(var.clone()) {
                    return Err(format!("Doubly controlled {}", var.id()));
                }
            }

            for lc in atom.local() {
                if wires.contains(lc) {
                    return Err(format!("local wire {} is also a module wire", lc.id()));
                }
                if !local.insert(lc.clone()) {
                    return Err(format!("local wire {} coupled with other atom", lc.id()));
                }
            }

            wires.extend(atom.wires().iter().cloned());

            for past_atom in past_atoms.iter() {
                if past_atom.awaits(&atom) {
                    return Err(format!("{}-th atom is in inconsistent awaiting order", n));
                }
            }
            past_atoms.push(atom);
        }

        let ctrl = Interface::from_exact_iter_unchecked(intf.iter().cloned());

        let prvt = intf.extract_if(.., &hide).collect::<Vec<_>>();
        for var in extl.iter() {
            if hide(var) {
                return Err(format!("Hiding external variable {}", var.id()));
            }
        }

        let mut obs: Vec<_> = extl.iter().chain(intf.iter()).cloned().collect();
        obs.sort_unstable();

        let extl = Interface::from_exact_iter_unchecked(extl);
        let intf = Interface::from_exact_iter_unchecked(intf);
        let prvt = Interface::from_exact_iter_unchecked(prvt);
        let obs = Interface::from_exact_iter_unchecked(obs);
        let wires = wires.into_iter().collect();
        let local = local.into_iter().collect();

        Ok(Self::new_unchecked(
            extl, intf, prvt, obs, ctrl, past_atoms, wires, local,
        ))
    }

    /// Constructs the **pure parallel composition** of several modules.
    ///
    /// This function takes an iterator of modules and returns a new module
    /// that represents all of them composed in parallel, coupling all shared
    /// observable variables. It is the special case of the *hiding
    /// composition* [`Module::hiding_composition`] that hides nothing: every observable
    /// of the components stays observable in the composite.
    ///
    /// # Semantics
    ///
    /// Observable variables shared across modules are *coupled* in the
    /// composed module: they represent the same value in the resulting
    /// system.
    ///
    /// # Error Conditions
    ///
    /// - A module attempts to couple a *private* or *temporary* wire with another module
    /// - A coupled variable is *controlled by more than one module*
    /// - Await dependency is cyclic
    ///
    /// # Returns
    ///
    /// - `Ok(Module)` containing the composed module.
    /// - `Err(Error)` describing the reason composition failed.
    ///
    /// # See Also
    /// - [`Module::hiding_composition`], the general form that also hides variables.
    pub fn composition<M>(modules: M) -> Result<Self, String>
    where
        M: IntoIterator<Item = Self>,
    {
        Self::hiding_composition(modules, |_| false)
    }

    /// The **hiding operator**: hides the given variables of this module.
    ///
    /// Consumes the module and returns one with the same behaviour in which
    /// the `hide` variables have left the observables and become private.
    /// It is the hiding composition [`Module::hiding_composition`] of the
    /// module alone, and composes with it: hiding after composing equals
    /// hiding at composition.
    ///
    /// Only interface variables can be hidden: hiding an external variable
    /// is rejected, since privates must be controlled.
    ///
    /// # Parameters
    /// - `hide`: The variables to hide; hiding nothing returns the module unchanged.
    ///
    /// # Returns
    ///
    /// - `Ok(Module)` containing the module with the variables hidden.
    /// - `Err(Error)` describing the reason hiding failed.
    ///
    /// # See Also
    /// - [`Module::hiding_composition`], hiding several modules at their composition.
    pub fn hiding<H>(self, hide: H) -> Result<Self, String>
    where
        H: Fn(&Var<S>) -> bool,
    {
        Self::hiding_composition(std::iter::once(self), hide)
    }

    /// Constructs the **hiding composition** of several modules: parallel
    /// composition that simultaneously hides the given variables.
    ///
    /// The modules are composed as in [`Module::composition`], coupling all shared
    /// observable variables; the `hide` variables are then removed from the
    /// composite's observables and become private. The coupling still takes
    /// place — hiding restricts the visibility of the *composite*, not the
    /// communication between the components.
    ///
    /// Only variables controlled by one of the components can be hidden:
    /// hiding an external variable is rejected, since privates must be
    /// controlled.
    ///
    /// # Parameters
    /// - `modules`: The modules to compose.
    /// - `hide`: The variables to hide in the composite; hiding nothing is
    ///   exactly [`Module::composition`].
    ///
    /// # Error Conditions
    ///
    /// All error conditions of [`Module::composition`], and additionally:
    ///
    /// - A prvt variable is external to the composition
    ///
    /// # Returns
    ///
    /// - `Ok(Module)` containing the composed module.
    /// - `Err(Error)` describing the reason composition failed.
    pub fn hiding_composition<M, H>(modules: M, hide: H) -> Result<Self, String>
    where
        M: IntoIterator<Item = Self>,
        H: Fn(&Var<S>) -> bool,
    {
        let mut declared_wires: BTreeSet<Wire<S>> = BTreeSet::new();
        let mut restricted_wires: HashSet<usize> = HashSet::new();

        let mut extl_set: BTreeSet<Var<S>> = BTreeSet::new();
        let mut intf_prior_hiding: HashSet<usize> = HashSet::new();

        let mut intf_stack: Vec<Var<S>> = Vec::new();
        let mut prvt_stack: Vec<Var<S>> = Vec::new();
        let mut obs_stack: Vec<Var<S>> = Vec::new();
        let mut ctrl_stack: Vec<Var<S>> = Vec::new();
        let mut local_stack: Vec<Wire<S>> = Vec::new();
        let mut atoms_stack: Vec<Atom<I, J, F, S>> = Vec::new();

        let mut await_graph: Vec<Vec<usize>> = Vec::new();

        for module in modules {
            //============================================================
            // Ensure decoupling and restrict visibility
            //============================================================

            // Check that observables are either uncoupled or coupled in right direction
            obs_stack.reserve(module.obs.len());
            for var in module.obs {
                for wire in var.wires() {
                    if restricted_wires.contains(&wire.id()) {
                        return Err(format!("coupling on restricted wire {}", wire.id()));
                    }
                }

                // visit every observable no more than once, and skip otherwise
                if !declared_wires.contains(var.ltc()) {
                    debug_assert!(!declared_wires.contains(var.nxt()));
                    debug_assert!(!declared_wires.contains(var.der()));
                    // hidden variables leave the observables and join the privates;
                    // we check at the end whether this has affected uncoupled externals
                    if !hide(&var) {
                        obs_stack.push(var);
                    } else {
                        prvt_stack.push(var);
                    }
                } else {
                    debug_assert!(declared_wires.contains(var.nxt()));
                    debug_assert!(declared_wires.contains(var.der()));
                }
            }

            // Check that privates are uncoupled and restrict them
            prvt_stack.reserve(module.prvt.len());
            for var in module.prvt {
                // visit every private no more than once, and raise otherwise
                if declared_wires.contains(var.ltc()) {
                    debug_assert!(declared_wires.contains(var.nxt()));
                    debug_assert!(declared_wires.contains(var.der()));
                    return Err(format!("private wire {} is declared elsewhere", var.id()));
                }
                debug_assert!(!declared_wires.contains(var.nxt()));
                debug_assert!(!declared_wires.contains(var.der()));

                for wire in var.wires() {
                    debug_assert!(!restricted_wires.contains(&wire.id()));
                    restricted_wires.insert(wire.id());
                }

                prvt_stack.push(var);
            }

            // Check that temporaries are uncoupled and restrict them
            local_stack.reserve(module.local.len());
            for tmp in module.local {
                // visit every local no more than once, and raise otherwise
                if declared_wires.contains(&tmp) {
                    return Err(format!("local wire {} is declared elsewhere", tmp.id()));
                }
                debug_assert!(!restricted_wires.contains(&tmp.id()));
                restricted_wires.insert(tmp.id());

                local_stack.push(tmp);
            }

            //============================================================
            // Couple external and interface variables
            //============================================================
            for var in module.extl {
                for wire in var.wires() {
                    debug_assert!(!restricted_wires.contains(&wire.id()));
                }

                // external variables stay external only if they are not
                // deemed controlled by modules visited before
                if !intf_prior_hiding.contains(&var.id()) {
                    extl_set.insert(var);
                }
            }

            intf_stack.reserve(module.intf.len());
            for var in module.intf {
                for wire in var.wires() {
                    debug_assert!(!restricted_wires.contains(&wire.id()));
                }

                // interface variables necessarily leave the set of variables
                // deemed external by modules visited before
                extl_set.remove(&var);

                if !intf_prior_hiding.insert(var.id()) {
                    return Err(format!("interface var {} is doubly controlled", var.id()));
                }

                if !prvt_stack.contains(&var) {
                    intf_stack.push(var);
                }
            }

            ctrl_stack.extend(module.ctrl);
            declared_wires.extend(module.wires);

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
        // Check that no hidden variables are externals
        //============================================================

        for var in extl_set.iter() {
            if hide(var) {
                return Err(format!("Cannot hide uncontrolled var {}", var.id()));
            }
        }

        //============================================================
        // Reorder interfaces
        //============================================================

        // post-reordering for efficient memory reservation above
        intf_stack.sort_unstable();
        ctrl_stack.sort_unstable();
        obs_stack.sort_unstable();
        prvt_stack.sort_unstable();
        local_stack.sort_unstable();

        //============================================================
        // Reorder atoms
        //============================================================

        let await_order = topological_order(&await_graph).ok_or("invalid await dependency")?;
        debug_assert_eq!(await_order.len(), await_graph.len());

        let mut atoms: Vec<Atom<I, J, F, S>> = Vec::with_capacity(await_graph.len());
        for idx in await_order {
            atoms.push(std::mem::take(&mut atoms_stack[idx]));
        }

        //============================================================
        // Collect and construct
        //============================================================

        let extl = Interface::from_exact_iter_unchecked(extl_set);
        let intf = Interface::from_exact_iter_unchecked(intf_stack);
        let prvt = Interface::from_exact_iter_unchecked(prvt_stack);
        let obs = Interface::from_exact_iter_unchecked(obs_stack);
        let ctrl = Interface::from_exact_iter_unchecked(ctrl_stack);
        let wires = declared_wires.into_iter().collect();

        Ok(Module::new_unchecked(
            extl,
            intf,
            prvt,
            obs,
            ctrl,
            atoms,
            wires,
            local_stack,
        ))
    }
}

//============================================================
// Display routines
//============================================================

pub struct Display<'a, I, J, F, S, N>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    module: &'a Module<I, J, F, S>,
    name: N,
}

impl<'a, I, J, F, S, N> Display<'a, I, J, F, S, N>
where
    I: Combinatorial<Sort = S> + fmt::Display,
    J: Sequential<Sort = S> + fmt::Display,
    F: Differential<Sort = S> + fmt::Display,
    S: fmt::Display,
    N: Fn(&Var<S>) -> Cow<'a, str>,
{
    fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, pad: &str) -> fmt::Result {
        const BOLD: &str = "\x1b[1m";
        const RESET: &str = "\x1b[0m";
        const INDENT: &str = "  ";
        const INDENT2: &str = "    ";

        let m = self.module;
        let name = &self.name;
        writeln!(f, "{pad}{BOLD}module{RESET}")?;
        if !m.extl.is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}external{RESET}")?;
        }
        for var in m.extl.iter() {
            writeln!(f, "{pad}{INDENT2}{} : {}", name(var), var.dtype())?;
        }
        if !m.intf.is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}interface{RESET}")?;
        }
        for var in m.intf.iter() {
            writeln!(f, "{pad}{INDENT2}{} : {}", name(var), var.dtype())?;
        }
        if !m.prvt.is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}private{RESET}")?;
        }
        for var in m.prvt.iter() {
            writeln!(f, "{pad}{INDENT2}{} : {}", name(var), var.dtype())?;
        }
        for atom in &m.atoms {
            atom.with_varnames_untyped(name)
                .fmt_indent(f, &format!("{pad}{INDENT}"))?;
        }
        Ok(())
    }
}

impl<'a, I, J, F, S, N> fmt::Display for Display<'a, I, J, F, S, N>
where
    I: Combinatorial<Sort = S> + fmt::Display,
    J: Sequential<Sort = S> + fmt::Display,
    F: Differential<Sort = S> + fmt::Display,
    S: fmt::Display,
    N: Fn(&Var<S>) -> Cow<'a, str>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, "")
    }
}

impl<I, J, D, S> Module<I, J, D, S>
where
    I: Combinatorial<Sort = S> + fmt::Display,
    J: Sequential<Sort = S> + fmt::Display,
    D: Differential<Sort = S> + fmt::Display,
    S: fmt::Display,
{
    /// Displays the module with variables under the names the given
    /// function assigns; it is consulted for every module variable.
    /// Displays the module with variables under the names the given
    /// function assigns; it is consulted for every module variable and may
    /// return anything that converts into a `Cow<'a, str>` -- `&'a str`,
    /// `String`, or a `Cow` itself.
    pub fn with_varnames<'a, N, R>(
        &'a self,
        name: N,
    ) -> Display<'a, I, J, D, S, impl Fn(&Var<S>) -> Cow<'a, str>>
    where
        N: Fn(&Var<S>) -> R,
        R: Into<Cow<'a, str>>,
    {
        Display {
            module: self,
            name: move |v: &Var<S>| name(v).into(),
        }
    }
}
