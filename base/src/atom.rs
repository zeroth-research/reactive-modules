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

    pub fn delay(&self) -> &Block<F> {
        &self.delay
    }

    pub fn ctrl(&self) -> &Interface<S> {
        &self.ctrl
    }

    pub fn wait(&self) -> &Interface<S> {
        &self.wait
    }

    pub fn read(&self) -> &Interface<S> {
        &self.read
    }

    pub fn temp(&self) -> impl Iterator<Item = &Wire<S>> {
        self.temp.iter()
    }

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
    /// Returns true if this atoms awaits the other atom
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

    /// Constructs a **sequential atom**, representing behaviour that evolves over time.
    ///
    /// A sequential atom defines both an initialisation (`init`) and an update (`update`)
    /// action. It relates *latched* (`current`) and *next* wires across discrete time steps.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are automatically
    /// inferred from the `[latched, next]` wire.
    /// This means the caller does not need to specify them manually.
    ///
    /// # Parameters
    /// - `latched`: The wire representing the latched variable.
    /// - `next`: The wire representing the next variable.
    /// - `init`: The terms defining the initial action of the atom.
    /// - `update`: The terms defining the state update at each time step.
    ///
    /// # Returns
    /// A `Result` containing the constructed sequential atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # Semantics
    /// Sequential atoms are **time-dependent**: they specify both an initial
    /// and a subsequent transition behaviour. This distinguishes them from
    /// [`combinatorial`] atoms, which are time-independent and purely reactive.
    ///
    /// # See Also
    /// - [`Atom::combinatorial`], for constructing combinatorial atoms.
    /// - [`Module::partially_observable_sequential`], for creating sequential modules.
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
        let ctrl_der = ctrl.values().map(Variable::nxt).cloned();

        let update = Block::skip(ctrl_nxt, ctrl_ltc)?;
        let delay = Block::zero(ctrl_der)?;

        Self::with_ctrl(wires, ctrl, init, update, delay)
    }

    /// Constructs a **purely combinatorial atom**, representing purely reactive behaviour
    /// without temporal state.
    ///
    /// A combinatorial atom defines a set of assignments (`assign`) that relate next
    /// wires *within the same time step*, without any notion of latching or sequential update.
    ///
    /// The control-related wires (`ctrl`, `wait`, `read`, and `temp`) are automatically
    /// inferred from the `next` wire and from the variables appearing in the input terms.
    /// This allows the caller to specify only the functional relationships,
    /// leaving wiring details to automatic inference.
    ///
    /// # Parameters
    /// - `next`: The output wire representing the combinatorial result.
    /// - `assign`: The terms defining the combinatorial relationships between next wires.
    ///
    /// # Returns
    /// A `Result` containing the constructed combinatorial atom if successful,
    /// or an error string if inference or consistency checks fail.
    ///
    /// # Semantics
    /// Combinatorial atoms are **time-independent**: they relate variables
    /// purely as functions of current inputs. They serve as the dual of
    /// [`sequential`] atoms, which include explicit time evolution.
    ///
    /// # See Also
    /// - [`Atom::sequential`], for constructing sequential atoms.
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
    /// - [`Module::partially_observable_sequential`], for constructing stateful, sequential modules.
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
