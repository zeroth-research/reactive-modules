use crate::Module;
use crate::term::{Block, Term};
use crate::wire::Wire;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use theory::{Combinatorial, Differential, Sequential, Theory};

use crate::variable::{Interface, Variable};
#[cfg(debug_assertions)]
use std::collections::HashSet;
use std::fmt::Debug;

/// This data structure corresponds to the atom of reactive modules.
#[derive(Debug, Clone)]
pub struct Atom<I, J, F, S = <I as Theory>::Sort>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    /// Corresponds to ctr wires.
    ctrl: Interface<S>,
    /// Corresponds to wait wires.
    wait: Interface<S>,
    /// Corresponds to read wires.
    read: Interface<S>,
    /// Corresponds to the initial condition.
    init: Block<I>,
    /// Corresponds to the update action.
    update: Block<J>,
    /// Corresponds to the delay activity.
    delay: Block<F>,

    /// Corresponds to temporary, local wires.
    temp: Vec<Wire<S>>,
}

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    /// Returns a reference to the initial action.
    pub fn init(&self) -> &Block<I> {
        &self.init
    }
    /// Returns a reference to the update action.
    pub fn update(&self) -> &Block<J> {
        &self.update
    }

    /// Returns a reference to the delay activity.
    pub fn delay(&self) -> &Block<F> {
        &self.delay
    }

    /// Returns a reference to the controlled variables.
    pub fn ctrl(&self) -> &Interface<S> {
        &self.ctrl
    }

    /// Returns a reference to the awaited variables.
    pub fn wait(&self) -> &Interface<S> {
        &self.wait
    }

    /// Returns a reference to the read variables.
    pub fn read(&self) -> &Interface<S> {
        &self.read
    }

    /// Returns an iterator over the temporary, local wires.
    pub fn temp(&self) -> impl Iterator<Item = &Wire<S>> {
        self.temp.iter()
    }

    /// Constructs an atom with no variables and empty blocks.
    pub fn empty() -> Self {
        Self {
            ctrl: Interface::empty(),
            wait: Interface::empty(),
            read: Interface::empty(),
            temp: Vec::new(),
            init: Block::empty(),
            update: Block::empty(),
            delay: Block::empty(),
        }
    }
}

impl<I, J, F, S> Default for Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    fn default() -> Self {
        Self::empty()
    }
}

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
{
    /// Returns true if this atom awaits the other atom.
    pub fn awaits(&self, other: &Atom<I, J, F, S>) -> bool {
        !self.wait.is_disjoint(&other.ctrl)
    }

    /// Creates an atom from its components. This method checks the inputs only using assertions
    /// in debug mode.
    fn new_unchecked(
        ctrl: Interface<S>,
        wait: Interface<S>,
        read: Interface<S>,
        temp: Vec<Wire<S>>,
        init: Block<I>,
        update: Block<J>,
        delay: Block<F>,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            //================================================================================
            // Check declared wires
            //================================================================================
            debug_assert!(ctrl.is_disjoint(&wait));

            //================================================================================
            // Check init terms
            //================================================================================
            // the init terms can initially read from the await wires of the atom
            let mut written: HashSet<&Wire<_>> = wait.iter().map(Variable::nxt).collect();
            for term in init.iter() {
                // all read wires were written before in the block
                debug_assert!(
                    term.read().iter().all(|rd| written.contains(rd)),
                    "read before write"
                );
                // no write wire was written before in the block
                debug_assert!(
                    term.write().iter().all(|rd| !written.contains(rd)),
                    "write after write"
                );
                written.extend(term.write().iter());
            }
            // all control wires are written
            debug_assert!(ctrl.iter().map(Variable::nxt).all(|w| written.contains(w)));

            //================================================================================
            // Check update terms
            //================================================================================
            // the update block can initially read from the read and await wires of the atom
            let mut written: HashSet<&Wire<_>> = read.iter().map(Variable::ltc).collect();
            written.extend(wait.iter().map(Variable::nxt));
            for term in update.iter() {
                // all read wires were written before in the block
                debug_assert!(
                    term.read().iter().all(|rd| written.contains(rd)),
                    "read before write"
                );
                // no write wire was written before in the block
                debug_assert!(
                    term.write().iter().all(|rd| !written.contains(rd)),
                    "write after write"
                );
                written.extend(term.write().iter());
            }
            // all control wires are written
            debug_assert!(ctrl.iter().map(Variable::nxt).all(|w| written.contains(w)));

            //================================================================================
            // Check delay terms
            //================================================================================
            // the delay block can initially read from the read and await wires of the atom
            let mut written: HashSet<&Wire<_>> = read.iter().map(Variable::ltc).collect();
            written.extend(wait.iter().map(Variable::der));
            for term in delay.iter() {
                // all read wires were written before in the block
                debug_assert!(
                    term.read().iter().all(|rd| written.contains(rd)),
                    "read before write"
                );
                // no write wire was written before in the block
                debug_assert!(
                    term.write().iter().all(|rd| !written.contains(rd)),
                    "write after write"
                );
                written.extend(term.write().iter());
            }
            // all control wires are written
            debug_assert!(ctrl.iter().map(Variable::der).all(|w| written.contains(w)));
        }

        Self {
            ctrl,
            wait,
            read,
            temp,
            init,
            update,
            delay,
        }
    }
}

struct WireMap<'a, S> {
    ltc: HashMap<&'a Wire<S>, &'a Variable<S>>,
    nxt: HashMap<&'a Wire<S>, &'a Variable<S>>,
    der: HashMap<&'a Wire<S>, &'a Variable<S>>,
}

impl<'a, S> WireMap<'a, S> {
    fn unpack<V>(vars: V) -> Self
    where
        V: IntoIterator<Item = &'a Variable<S>>,
    {
        let mut ltc: HashMap<&Wire<S>, &Variable<S>> = HashMap::new();
        let mut nxt: HashMap<&Wire<S>, &Variable<S>> = HashMap::new();
        let mut der: HashMap<&Wire<S>, &Variable<S>> = HashMap::new();

        for var in vars.into_iter() {
            ltc.insert(var.ltc(), var);
            nxt.insert(var.nxt(), var);
            der.insert(var.der(), var);
        }

        Self { ltc, nxt, der }
    }
}

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Clone + Debug + Eq,
{
    fn infer_ctrl<T>(
        pool: &HashMap<&Wire<S>, &Variable<S>>,
        block: &Block<T>,
    ) -> Result<BTreeMap<usize, Variable<S>>, String>
    where
        T: Theory<Sort = S>,
    {
        // tree map on id is used to guarantee consistent order
        let mut ctrl: BTreeMap<usize, Variable<T::Sort>> = BTreeMap::new();

        for wt in block.write().iter() {
            // if the block writes to a wire in the pool of next or derived,
            // then the respective variable is controlled
            if let Some(&var) = pool.get(&wt) {
                ctrl.insert(var.id(), var.clone());
            }
        }

        Ok(ctrl)
    }

    fn with_ctrl(
        wires: WireMap<S>,
        ctrl: BTreeMap<usize, Variable<S>>,
        init: Block<I>,
        update: Block<J>,
        delay: Block<F>,
    ) -> Result<Self, String> {
        // tree map on id is used to guarantee consistent order
        let mut read: BTreeMap<usize, Variable<S>> = BTreeMap::new();
        let mut wait: BTreeMap<usize, Variable<S>> = BTreeMap::new();
        let mut local: BTreeMap<usize, Wire<S>> = BTreeMap::new();

        for rd in init.read().iter() {
            // init can only read from next wires
            if let Some(&var) = wires.nxt.get(&rd) {
                wait.insert(var.id(), var.clone());
                continue;
            }

            if wires.ltc.contains_key(&rd) {
                return Err(format!("Init reads latched wire {}", rd.id()));
            }

            // dangling read wires are invalid
            return Err(format!("Wire {} in init is dangling read", rd.id()));
        }

        for rd in update.read().iter() {
            // if the update reads from a next wire, then this is awaited
            // otherwise, this must be read from outside the atom
            if let Some(&var) = wires.ltc.get(&rd) {
                read.insert(var.id(), var.clone());
                continue;
            }

            if let Some(&var) = wires.nxt.get(&rd) {
                wait.insert(var.id(), var.clone());
                continue;
            }

            // dangling read wires are parameters
            return Err(format!("Wire {} in update is dangling read", rd.id()));
        }

        for rd in delay.read().iter() {
            // if the delay reads from a derived wire, then this is awaited
            // otherwise, this must be read from outside the atom
            if let Some(&var) = wires.ltc.get(&rd) {
                read.insert(var.id(), var.clone());
                continue;
            }

            if let Some(&var) = wires.der.get(&rd) {
                wait.insert(var.id(), var.clone());
                continue;
            }

            // dangling read wires are parameters
            return Err(format!("Wire {} in update is dangling read", rd.id()));
        }

        for wt in init.write().iter().chain(update.write().iter()) {
            // if the init/update writes to a next wire, then this wire must be controlled
            // otherwise, this wire must be temporary
            if let Some(&var) = wires.nxt.get(&wt) {
                if !ctrl.contains_key(&var.id()) {
                    return Err(format!("Inconsistent write to next wire {}", wt.id()));
                }
                continue;
            }

            if wires.ltc.contains_key(&wt) {
                return Err(format!("Writing latched wire {}", wt.id()));
            } else {
                local.insert(wt.id(), wt.clone());
            }
        }

        for wt in delay.write().iter() {
            // if the init/update writes to a next wire, then this wire is controlled
            // otherwise, this wire must be temporary
            if let Some(&var) = wires.der.get(&wt) {
                if !ctrl.contains_key(&var.id()) {
                    return Err(format!("Inconsistent write to next wire {}", wt.id()));
                }
                continue;
            }

            if wires.ltc.contains_key(&wt) {
                return Err(format!("Writing a latched wire {}", wt.id()));
            } else {
                local.insert(wt.id(), wt.clone());
            }
        }

        for ctr in ctrl.values() {
            if !init.write().iter().any(|wrt| wrt == ctr.nxt()) {
                return Err(format!(
                    "Controlled wire {} is not written in init",
                    ctr.nxt().id()
                ));
            }
            if !update.write().iter().any(|wrt| wrt == ctr.nxt()) {
                return Err(format!(
                    "Controlled wire {} is not written in update",
                    ctr.der().id()
                ));
            }
            if !delay.write().iter().any(|wrt| wrt == ctr.der()) {
                return Err(format!(
                    "Controlled wire {} is not written in delay",
                    ctr.der().id()
                ));
            }
        }

        Ok(Self::new_unchecked(
            Interface::from_iter_unchecked(ctrl.into_values()),
            Interface::from_iter_unchecked(wait.into_values()),
            Interface::from_iter_unchecked(read.into_values()),
            local.into_values().collect(),
            init,
            update,
            delay,
        ))
    }

    /// Constructs a **sequential atom**: behaviour that evolves over discrete
    /// time steps.
    ///
    /// A sequential atom specifies an initialisation (`init`) and a discrete
    /// state update (`update`), relating the latched wires to the next wires
    /// across time steps. The delay is synthesised as `ZERO`, so the atom has
    /// no continuous dynamics.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are
    /// inferred from the variables and the terms; the controlled variables
    /// are those whose next wire is written by `init`.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `init`: The terms defining the initial action of the atom.
    /// - `update`: The terms defining the discrete state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed sequential atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::differential`], the continuous dual: only a delay, implicit skip update.
    /// - [`Atom::hybrid`], when both discrete and continuous dynamics are explicit.
    /// - [`Atom::combinatorial`], for time-independent, purely reactive behaviour.
    pub fn sequential<'a, V, W, U>(vars: V, init: W, update: U) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Variable<S>>,
        W: IntoIterator<Item = Term<I>>,
        U: IntoIterator<Item = Term<J>>,
        S: 'a + fmt::Display,
    {
        let wires = WireMap::unpack(vars);

        let init = Block::try_from_iter(init.into_iter().map(Ok))?;
        let ctrl = Self::infer_ctrl(&wires.nxt, &init)?;
        let ctrl_der = ctrl.values().map(Variable::der).cloned();

        let update = Block::try_from_iter(update.into_iter().map(Ok))?;
        let delay = Block::zero(ctrl_der)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **differential atom**: behaviour that evolves continuously.
    ///
    /// A differential atom specifies an initialisation (`init`) and a
    /// continuous evolution (`delay`) driving the derivative wires. The
    /// discrete update is synthesised as one `SKIP` term per controlled
    /// variable, copying the latched value to the next wire, so all change
    /// comes from the continuous dynamics.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are
    /// inferred from the variables and the terms; the controlled variables
    /// are those whose next wire is written by `init`.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `init`: The terms defining the initial action of the atom.
    /// - `delay`: The terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed differential atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::sequential`], the discrete dual: only an update, zero delay.
    /// - [`Atom::hybrid`], when both discrete and continuous dynamics are explicit.
    pub fn differential<'a, V, W, Z>(vars: V, init: W, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Variable<S>>,
        W: IntoIterator<Item = Term<I>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars);

        let init = Block::try_from_iter(init.into_iter().map(Ok))?;
        let ctrl = Self::infer_ctrl(&wires.nxt, &init)?;
        let ctrl_nxt = ctrl.values().map(Variable::nxt).cloned();
        let ctrl_ltc = ctrl.values().map(Variable::ltc).cloned();

        let update = Block::skip(ctrl_nxt, ctrl_ltc)?;
        let delay = Block::try_from_iter(delay.into_iter().map(Ok))?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **hybrid atom**, combining discrete and continuous dynamics.
    ///
    /// A hybrid atom specifies all three blocks explicitly: an initialisation
    /// (`init`), a discrete state update (`update`), and a continuous
    /// evolution (`delay`) driving the derivative wires. It is the most
    /// general constructor; [`Atom::sequential`] and [`Atom::differential`]
    /// are the special cases where the delay (resp. the update) is
    /// synthesised automatically.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are
    /// inferred from the variables and the terms; the controlled variables
    /// are those whose next wire is written by `init`.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `init`: The terms defining the initial action of the atom.
    /// - `update`: The terms defining the discrete state update at each time step.
    /// - `delay`: The terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed hybrid atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::sequential`], for purely discrete behaviour (delay is zero).
    /// - [`Atom::differential`], for purely continuous behaviour (update is skip).
    pub fn hybrid<'a, V, W, U, Z>(vars: V, init: W, update: U, delay: Z) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Variable<S>>,
        W: IntoIterator<Item = Term<I>>,
        U: IntoIterator<Item = Term<J>>,
        Z: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars);

        let init = Block::try_from_iter(init.into_iter().map(Ok))?;
        let update = Block::try_from_iter(update.into_iter().map(Ok))?;
        let delay = Block::try_from_iter(delay.into_iter().map(Ok))?;

        let ctrl = Self::infer_ctrl(&wires.nxt, &init)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs an **uninitialized atom**: sequential behaviour whose
    /// initial state is left unconstrained.
    ///
    /// Only the update is given; the controlled variables are inferred from
    /// the next wires it writes. The initialisation is synthesised as one
    /// `HAVOC` term per controlled variable, so their initial values are
    /// arbitrary, and the delay is synthesised as `ZERO`, so the atom has no
    /// continuous dynamics.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `update`: The terms defining the discrete state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::sequential`], when the initial state is constrained explicitly.
    /// - [`Atom::constant`], the dual: only the initial state, no update.
    pub fn uninitialized<'a, V, U>(vars: V, update: U) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Variable<S>>,
        U: IntoIterator<Item = Term<J>>,
        S: 'a + fmt::Display,
    {
        let wires = WireMap::unpack(vars);

        let update = Block::try_from_iter(update.into_iter().map(Ok))?;
        let ctrl = Self::infer_ctrl(&wires.nxt, &update)?;
        let ctrl_nxt = ctrl.values().map(Variable::nxt).cloned();
        let ctrl_der = ctrl.values().map(Variable::der).cloned();

        let init = Block::havoc(ctrl_nxt)?;
        let delay = Block::zero(ctrl_der)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **constant atom**: variables that are set once at
    /// initialisation and then hold their value forever.
    ///
    /// Only the initialisation is given; the controlled variables are
    /// inferred from the next wires it writes. The update is synthesised as
    /// one `SKIP` term per controlled variable, copying the latched value to
    /// the next wire at every step, and the delay is synthesised as `ZERO`,
    /// so the values never drift.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `init`: The terms defining the (only) assignment of the variables.
    ///
    /// # Returns
    /// A `Result` containing the constructed atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::uninitialized`], the dual: only an update, arbitrary initial state.
    /// - [`Atom::sequential`], when the variables also evolve over time.
    pub fn constant<'a, V, W>(vars: V, init: W) -> Result<Self, String>
    where
        V: IntoIterator<Item = &'a Variable<S>>,
        W: IntoIterator<Item = Term<I>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars);

        let init = Block::try_from_iter(init.into_iter().map(Ok))?;
        let ctrl = Self::infer_ctrl(&wires.nxt, &init)?;
        let ctrl_ltc = ctrl.values().map(Variable::ltc).cloned();
        let ctrl_nxt = ctrl.values().map(Variable::nxt).cloned();
        let ctrl_der = ctrl.values().map(Variable::der).cloned();

        let update = Block::skip(ctrl_nxt, ctrl_ltc)?;
        let delay = Block::zero(ctrl_der)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **combinatorial atom**: purely reactive behaviour without
    /// temporal state.
    ///
    /// A combinatorial atom specifies a single block of assignments (`assign`)
    /// relating next wires *within the same time step*. The same block serves
    /// as both initialisation and update, and the delay is synthesised as
    /// `ZERO`, so the atom is time-independent.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are
    /// inferred from the variables and the terms; the controlled variables
    /// are those whose next wire is written by `assign`.
    ///
    /// # Parameters
    /// - `vars`: The variables in scope for the atom.
    /// - `assign`: The terms defining the combinatorial relationships between next wires.
    ///
    /// # Returns
    /// A `Result` containing the constructed combinatorial atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Atom::sequential`], the stateful dual with explicit time evolution.
    /// - [`Module::combinatorial`], for combinatorial modules.
    pub fn combinatorial<'a, T, V, W>(vars: V, assign: W) -> Result<Self, String>
    where
        T: Theory<Sort = S> + Into<I> + Into<J> + Clone,
        V: IntoIterator<Item = &'a Variable<S>>,
        W: IntoIterator<Item = Term<T>>,
        S: 'a,
    {
        let wires = WireMap::unpack(vars);

        let assign: Block<T> = Block::try_from_iter(assign.into_iter().map(Ok))?;
        let ctrl = Self::infer_ctrl(&wires.nxt, &assign)?;

        let init: Block<I> = Block::try_from_iter(assign.iter().cloned().map(Ok))?;
        let update: Block<J> = Block::try_from_iter(assign.into_iter().map(Ok))?;
        let delay: Block<F> = Block::zero(ctrl.values().map(Variable::der).cloned())?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }
}

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S> + fmt::Display,
    J: Sequential<Sort = S> + fmt::Display,
    F: Differential<Sort = S> + fmt::Display,
    S: fmt::Display,
{
    pub(crate) fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, pad: &str) -> fmt::Result {
        const BOLD: &str = "\x1b[1m";
        const RESET: &str = "\x1b[0m";
        const INDENT: &str = "  ";

        write!(f, "{pad}{BOLD}atom{RESET}")?;
        for (i, v) in self.ctrl.iter().enumerate() {
            if i == 0 {
                write!(f, " {BOLD}controls{RESET} {v}")?;
            } else {
                write!(f, ", w{v}")?;
            }
        }
        for (i, v) in self.read.iter().enumerate() {
            if i == 0 {
                write!(f, " {BOLD}reads{RESET} {v}")?;
            } else {
                write!(f, ", w{v}")?;
            }
        }
        for (i, v) in self.wait.iter().enumerate() {
            if i == 0 {
                write!(f, " {BOLD}awaits{RESET} {v}")?;
            } else {
                write!(f, ", w{v}")?;
            }
        }
        writeln!(f, "\n{pad}{BOLD}init{RESET}")?;

        for term in self.init.iter() {
            writeln!(f, "{pad}{INDENT}{term}")?;
        }
        writeln!(f, "{pad}{BOLD}delay{RESET}")?;
        for term in self.delay.iter() {
            writeln!(f, "{pad}{INDENT}{term}")?;
        }
        writeln!(f, "{pad}{BOLD}update{RESET}")?;
        for term in self.update.iter() {
            writeln!(f, "{pad}{INDENT}{term}")?;
        }
        Ok(())
    }
}

impl<I, J, F, S> fmt::Display for Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S> + fmt::Display,
    J: Sequential<Sort = S> + fmt::Display,
    F: Differential<Sort = S> + fmt::Display,
    S: fmt::Display + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, "")
    }
}

impl<I, J, F, S> Module<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Eq + Clone + Debug + fmt::Display,
{
    /// Constructs a **sequential module**: behaviour that evolves over
    /// discrete time steps.
    ///
    /// A sequential module specifies an initialisation and a discrete state
    /// update; it has no continuous dynamics. It is composed of a single
    /// [`Atom::sequential`] atom, and is **fully observable by default**.
    ///
    /// # Parameters
    /// - `obs`: The sequence of variables representing the module's observables.
    /// - `init`: The set of terms defining the module's initial state.
    /// - `update`: The set of terms defining the module's state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed sequential module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::differential`] and [`Module::hybrid`], when continuous
    ///   dynamics are involved.
    /// - [`Module::combinatorial`], for stateless, time-independent modules.
    /// - [`Atom::sequential`], for creating individual sequential atoms.
    pub fn sequential<O, V, W, U>(obs: O, init: W, update: U) -> Result<Self, String>
    where
        V: Into<Variable<S>>,
        O: IntoIterator<Item = V>,
        W: IntoIterator<Item = Term<I>>,
        U: IntoIterator<Item = Term<J>>,
    {
        let obs: Vec<_> = obs.into_iter().map(Into::into).collect();
        let atom = Atom::sequential(obs.iter(), init, update)?;
        Self::observable(obs, std::iter::once(atom))
    }

    /// Constructs a **differential module**: behaviour that evolves
    /// continuously.
    ///
    /// A differential module specifies an initialisation and a continuous
    /// evolution of the derivatives; the discrete update is an implicit skip.
    /// It is composed of a single [`Atom::differential`] atom, and is
    /// **fully observable by default**.
    ///
    /// # Parameters
    /// - `obs`: The sequence of variables representing the module's observables.
    /// - `init`: The set of terms defining the module's initial state.
    /// - `delay`: The set of terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed differential module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::sequential`], the discrete dual: only an update, no continuous dynamics.
    /// - [`Module::hybrid`], when both discrete and continuous dynamics are explicit.
    /// - [`Atom::differential`], for creating individual differential atoms.
    pub fn differential<O, V, W, Z>(obs: O, init: W, delay: Z) -> Result<Self, String>
    where
        V: Into<Variable<S>>,
        O: IntoIterator<Item = V>,
        W: IntoIterator<Item = Term<I>>,
        Z: IntoIterator<Item = Term<F>>,
    {
        let obs = obs.into_iter().map(Into::into).collect::<Vec<_>>();
        let atom = Atom::differential(obs.iter(), init, delay)?;
        Self::observable(obs, std::iter::once(atom))
    }

    /// Constructs a **hybrid module**, combining discrete updates with
    /// continuous dynamics.
    ///
    /// A hybrid module specifies all three blocks explicitly — initialisation,
    /// discrete update, and continuous delay. It is composed of a single
    /// [`Atom::hybrid`] atom, and is **fully observable by default**.
    ///
    /// # Parameters
    /// - `obs`: The sequence of variables representing the module's observables.
    /// - `init`: The set of terms defining the module's initial state.
    /// - `update`: The set of terms defining the discrete state update.
    /// - `delay`: The set of terms defining the continuous evolution of the derivatives.
    ///
    /// # Returns
    /// A `Result` containing the constructed hybrid module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::sequential`] and [`Module::differential`], the purely
    ///   discrete and purely continuous special cases.
    /// - [`Atom::hybrid`], for creating individual hybrid atoms.
    pub fn hybrid<O, V, W, U, Z>(obs: O, init: W, update: U, delay: Z) -> Result<Self, String>
    where
        V: Into<Variable<S>>,
        O: IntoIterator<Item = V>,
        W: IntoIterator<Item = Term<I>>,
        U: IntoIterator<Item = Term<J>>,
        Z: IntoIterator<Item = Term<F>>,
    {
        let obs = obs.into_iter().map(Into::into).collect::<Vec<_>>();
        let atom = Atom::hybrid(obs.iter(), init, update, delay)?;
        Self::observable(obs, std::iter::once(atom))
    }

    /// Constructs an **uninitialized module**: sequential behaviour whose
    /// initial state is left unconstrained.
    ///
    /// Only the update is given; the initial values of the controlled
    /// variables are havoced. It is composed of a single
    /// [`Atom::uninitialized`] atom, and is **fully observable by default**.
    ///
    /// # Parameters
    /// - `obs`: The sequence of variables representing the module's observables.
    /// - `update`: The set of terms defining the module's state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::sequential`], when the initial state is constrained explicitly.
    /// - [`Module::constant`], the dual: only the initial state, no update.
    pub fn uninitialized<O, V, U>(obs: O, update: U) -> Result<Self, String>
    where
        V: Into<Variable<S>>,
        O: IntoIterator<Item = V>,
        U: IntoIterator<Item = Term<J>>,
    {
        let obs = obs.into_iter().map(Into::into).collect::<Vec<_>>();
        let atom = Atom::uninitialized(obs.iter(), update)?;
        Self::observable(obs, std::iter::once(atom))
    }

    /// Constructs a **constant module**: variables that are set once at
    /// initialisation and then hold their value forever.
    ///
    /// Only the initialisation is given; the variables are held by an implicit
    /// skip update and have no continuous dynamics. It is composed of a single
    /// [`Atom::constant`] atom, and is **fully observable by default**.
    ///
    /// # Parameters
    /// - `obs`: The sequence of variables representing the module's observables.
    /// - `init`: The set of terms defining the (only) assignment of the variables.
    ///
    /// # Returns
    /// A `Result` containing the constructed module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::uninitialized`], the dual: only an update, arbitrary initial state.
    /// - [`Atom::constant`], for creating individual constant atoms.
    pub fn constant<O, V, W>(obs: O, init: W) -> Result<Self, String>
    where
        V: Into<Variable<S>>,
        O: IntoIterator<Item = V>,
        W: IntoIterator<Item = Term<I>>,
    {
        let obs = obs.into_iter().map(Into::into).collect::<Vec<_>>();
        let atom = Atom::constant(obs.iter(), init)?;
        Self::observable(obs, std::iter::once(atom))
    }

    /// Constructs a **combinatorial module**: a stateless, time-independent
    /// relationship between observable wires.
    ///
    /// A combinatorial module specifies a single block of assignments
    /// computing the outputs from the inputs within the same time step. It is
    /// composed of a single [`Atom::combinatorial`] atom, and is **fully
    /// observable by default**.
    ///
    /// # Parameters
    /// - `obs`: The sequence of variables representing the module's observables.
    /// - `assign`: The set of combinatorial assignment terms defining how the
    ///   output is computed from the input.
    ///
    /// # Returns
    /// A `Result` containing the constructed combinatorial module if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # See Also
    /// - [`Module::sequential`], the stateful dual with explicit time evolution.
    /// - [`Atom::combinatorial`], for creating individual combinatorial atoms.
    pub fn combinatorial<T, V, O, W>(obs: O, assign: W) -> Result<Self, String>
    where
        T: Theory<Sort = S> + Into<I> + Into<J> + Clone,
        V: Into<Variable<S>>,
        O: IntoIterator<Item = V>,
        W: IntoIterator<Item = Term<T>>,
    {
        let obs = obs.into_iter().map(Into::into).collect::<Vec<_>>();
        let atom = Atom::combinatorial(obs.iter(), assign)?;
        Self::observable(obs, [atom])
    }
}
