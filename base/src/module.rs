use crate::atom::Atom;
use crate::topological_order;
use crate::var::{Interface, Var};
use crate::wire::Wire;
use std::collections::{BTreeSet, HashSet};
use std::fmt;
use std::fmt::Debug;
use theory::{Combinatorial, Differential, Sequential, Theory};

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
    ///  Wires are organised in pairs of identical twins where
    ///  - 0: latched wires
    ///  - 1: next wires
    extl: Interface<S>,
    intf: Interface<S>,
    prvt: Interface<S>,
    obs: Interface<S>,
    ctrl: Interface<S>,

    /// The atoms of this module.
    /// The atoms must be stored in a *consistent* linear order
    /// as defined in the reactive modules paper.
    atoms: Vec<Atom<I, J, F, S>>,

    temp: Vec<Wire<S>>,
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

    pub fn temp(&self) -> impl Iterator<Item = &Wire<S>> {
        self.temp.iter()
    }

    pub fn empty() -> Self {
        Module {
            extl: Interface::empty(),
            intf: Interface::empty(),
            prvt: Interface::empty(),
            obs: Interface::empty(),
            ctrl: Interface::empty(),
            temp: Vec::new(),
            atoms: Vec::new(),
        }
    }
}

impl<I, J, F, S> Module<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
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
        temp: Vec<Wire<S>>,
        atoms: Vec<Atom<I, J, F, S>>,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(obs.len(), extl.len() + intf.len());
            debug_assert_eq!(ctrl.len(), intf.len() + prvt.len());

            let mut decl: HashSet<usize> = HashSet::new();

            debug_assert!(extl.iter().all(|v| decl.insert(v.id())));
            let extl: HashSet<usize> = extl.iter().map(|v| v.id()).collect();

            debug_assert!(intf.iter().all(|v| decl.insert(v.id())));
            let intf: HashSet<usize> = intf.iter().map(|v| v.id()).collect();

            debug_assert!(prvt.iter().all(|v| decl.insert(v.id())));
            let prvt: HashSet<usize> = prvt.iter().map(|v| v.id()).collect();

            debug_assert!(obs.iter().all(|v| !decl.insert(v.id())));
            debug_assert!(
                obs.iter()
                    .all(|v| extl.contains(&v.id()) || intf.contains(&v.id()))
            );

            debug_assert!(ctrl.iter().all(|v| !decl.insert(v.id())));
            debug_assert!(
                ctrl.iter()
                    .all(|v| intf.contains(&v.id()) || prvt.contains(&v.id()))
            );

            let mut written: HashSet<usize> = HashSet::new();
            written.extend(extl.iter());
            // check atoms consistency
            for atom in atoms.iter() {
                for var in atom.read().iter() {
                    debug_assert!(decl.contains(&var.id()));
                }
                for var in atom.wait().iter() {
                    debug_assert!(decl.contains(&var.id()));
                    debug_assert!(written.contains(&var.id()));
                }
                for var in atom.ctrl().iter() {
                    debug_assert!(decl.contains(&var.id()));
                    debug_assert!(written.insert(var.id()));
                }
            }

            // check that all module control vars are written/controlled by an atom
            for var in ctrl.iter() {
                debug_assert!(written.contains(&var.id()));
            }

            // check that temporaries are decoupled from module wires and other atoms
            let mut module_temp: HashSet<usize> = HashSet::new();
            for lc in atoms.iter().flat_map(Atom::temp) {
                debug_assert!(temp.contains(lc));
                debug_assert!(module_temp.insert(lc.id()));
            }
            debug_assert_eq!(module_temp.len(), temp.len());
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
}

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
        Self::partially_observable(atoms, std::iter::empty())
    }

    /// Constructs a **partially observable module** from a sequence of atoms,
    /// hiding the given variables.
    ///
    /// The variables and their roles are inferred from the atoms — the
    /// controlled variables form the interface, the remaining read or awaited
    /// variables are external — and the `hidden` variables are then moved
    /// from the interface into the private state. Only controlled variables
    /// can be hidden: privates must be controlled by the module.
    ///
    /// # Parameters
    /// - `atoms`: An iterable collection of atoms defining the module's behaviour.
    /// - `hidden`: The variables to hide from the environment; hiding nothing
    ///   yields a fully observable module.
    ///
    /// # Returns
    /// A `Result` containing the constructed partially observable module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::observable`], the fully observable special case.
    /// - [`Atom::sequential`], [`Atom::combinatorial`] for creating individual atoms.
    pub fn partially_observable<'a, A, H>(atoms: A, hidden: H) -> Result<Self, String>
    where
        A: IntoIterator<Item = Atom<I, J, F, S>> + Sized,
        H: IntoIterator<Item = &'a Var<S>>,
        S: 'a,
    {
        let mut wires: HashSet<Wire<S>> = HashSet::new();
        let mut local: HashSet<Wire<S>> = HashSet::new();

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

            for lc in atom.temp() {
                if wires.contains(&lc) {
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

        let ctrl = Interface::from_iter_unchecked(intf.iter().cloned());

        let mut prvt: BTreeSet<Var<S>> = BTreeSet::new();
        for var in hidden {
            if let Some(var) = intf.take(var) {
                prvt.insert(var);
            }
            if extl.contains(var) {
                return Err(format!("Hiding external variable {}", var.id()));
            }
        }

        let mut obs: Vec<_> = extl.iter().chain(intf.iter()).cloned().collect();
        obs.sort_unstable();

        let extl = Interface::from_iter_unchecked(extl);
        let intf = Interface::from_iter_unchecked(intf);
        let prvt = Interface::from_iter_unchecked(prvt);
        let obs = Interface::from_iter_unchecked(obs);
        let temp = local.into_iter().collect();

        Ok(Self::new_unchecked(
            extl, intf, prvt, obs, ctrl, temp, past_atoms,
        ))
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
    pub fn parallel<M>(modules: M) -> Result<Self, String>
    where
        M: IntoIterator<Item = Self>,
    {
        let mut observable_wire: HashSet<usize> = HashSet::new();
        let mut restricted_wire: HashSet<usize> = HashSet::new();

        let mut extl_set: BTreeSet<Var<S>> = BTreeSet::new();
        let mut intf_ids: HashSet<usize> = HashSet::new();

        let mut intf_stack: Vec<Var<S>> = Vec::new();
        let mut prvt_stack: Vec<Var<S>> = Vec::new();
        let mut obs_stack: Vec<Var<S>> = Vec::new();
        let mut ctrl_stack: Vec<Var<S>> = Vec::new();
        let mut temp_stack: Vec<Wire<S>> = Vec::new();
        let mut atoms_stack: Vec<Atom<I, J, F, S>> = Vec::new();

        let mut await_graph: Vec<Vec<usize>> = Vec::new();

        for module in modules {
            //============================================================
            // Ensure decoupling and restrict visibility
            //============================================================

            // Check that observables are either uncoupled or coupled in right direction
            obs_stack.reserve(module.obs.len());
            for var in module.obs {
                if restricted_wire.contains(&var.ltc().id()) {
                    return Err(format!("wire {} is restricted, got observable", var.id()));
                }
                if restricted_wire.contains(&var.nxt().id()) {
                    return Err(format!("wire {} is restricted, got observable", var.id()));
                }
                if restricted_wire.contains(&var.der().id()) {
                    return Err(format!("wire {} is restricted, got observable", var.id()));
                }

                if observable_wire.insert(var.ltc().id()) {
                    observable_wire.insert(var.nxt().id());
                    observable_wire.insert(var.der().id());
                    obs_stack.push(var);
                }
            }

            // Check that privates are uncoupled and restrict them
            prvt_stack.reserve(module.prvt.len());
            for var in module.prvt {
                if observable_wire.contains(&var.ltc().id()) {
                    debug_assert!(observable_wire.contains(&var.nxt().id()));
                    debug_assert!(observable_wire.contains(&var.der().id()));
                    return Err(format!(
                        "var {} is private, but observable elsewhere",
                        var.id()
                    ));
                }

                debug_assert!(!restricted_wire.contains(&var.ltc().id()));
                debug_assert!(!restricted_wire.contains(&var.nxt().id()));
                debug_assert!(!restricted_wire.contains(&var.der().id()));
                restricted_wire.insert(var.ltc().id());
                restricted_wire.insert(var.nxt().id());
                restricted_wire.insert(var.der().id());

                prvt_stack.push(var);
            }

            // Check that temporaries are uncoupled and restrict them
            temp_stack.reserve(module.temp.len());
            for tmp in module.temp {
                if observable_wire.contains(&tmp.id()) {
                    return Err(format!("local wire {} is observable elsewhere", tmp.id()));
                }
                if !restricted_wire.insert(tmp.id()) {
                    return Err(format!("local wire {} is restricted elsewhere", tmp.id()));
                }

                temp_stack.push(tmp);
            }

            //============================================================
            // Couple external and interface variables
            //============================================================
            for var in module.extl {
                if restricted_wire.contains(&var.ltc().id()) {
                    return Err(format!("wire {} is restricted, got external", var.id()));
                }
                if restricted_wire.contains(&var.nxt().id()) {
                    return Err(format!("wire {} is restricted, got external", var.id()));
                }
                if restricted_wire.contains(&var.der().id()) {
                    return Err(format!("wire {} is restricted, got external", var.id()));
                }

                if !intf_ids.contains(&var.id()) {
                    extl_set.insert(var);
                }
            }

            intf_stack.reserve(module.intf.len());
            for var in module.intf {
                if restricted_wire.contains(&var.ltc().id()) {
                    return Err(format!("wire {} is restricted, got external", var.id()));
                }
                if restricted_wire.contains(&var.nxt().id()) {
                    return Err(format!("wire {} is restricted, got external", var.id()));
                }
                if restricted_wire.contains(&var.der().id()) {
                    return Err(format!("wire {} is restricted, got external", var.id()));
                }

                extl_set.remove(&var);

                if !intf_ids.insert(var.id()) {
                    return Err(format!("interface var {} is doubly controlled", var.id()));
                }

                intf_stack.push(var);
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
        // Reorder interfaces
        //============================================================

        // post-reordering for efficient memory reservation above
        intf_stack.sort_unstable();
        ctrl_stack.sort_unstable();
        obs_stack.sort_unstable();
        ctrl_stack.sort_unstable();

        //============================================================
        // Reorder atoms and remove coupled wires from the externals
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

        let extl = Interface::from_iter_unchecked(extl_set);
        let intf = Interface::from_iter_unchecked(intf_stack);
        let prvt = Interface::from_iter_unchecked(prvt_stack);
        let obs = Interface::from_iter_unchecked(obs_stack);
        let ctrl = Interface::from_iter_unchecked(ctrl_stack);

        Ok(Module::new_unchecked(
            extl, intf, prvt, obs, ctrl, temp_stack, atoms,
        ))
    }
}

impl<I, J, D, S> Module<I, J, D, S>
where
    I: Combinatorial<Sort = S> + fmt::Display,
    J: Sequential<Sort = S> + fmt::Display,
    D: Differential<Sort = S> + fmt::Display,
    S: fmt::Display,
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
        for var in self.extl.iter() {
            writeln!(f, "{pad}{INDENT2}{var}")?;
        }
        if !self.intf.is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}interface{RESET}")?;
        }
        for var in self.intf.iter() {
            writeln!(f, "{pad}{INDENT2}{var}")?;
        }
        if !self.prvt.is_empty() {
            writeln!(f, "{pad}{INDENT}{BOLD}private{RESET}")?;
        }
        for var in self.prvt.iter() {
            writeln!(f, "{pad}{INDENT2}{var}")?;
        }
        for atom in &self.atoms {
            atom.fmt_indent(f, &format!("{pad}{INDENT}"))?;
        }
        Ok(())
    }
}

impl<I, J, D, S> fmt::Display for Module<I, J, D, S>
where
    I: Combinatorial<Sort = S> + fmt::Display,
    J: Sequential<Sort = S> + fmt::Display,
    D: Differential<Sort = S> + fmt::Display,
    S: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, "")
    }
}
