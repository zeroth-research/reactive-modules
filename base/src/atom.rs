use crate::Module;
use crate::term::{Block, Term};
use crate::wire::{Interface, Wire};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use theory::{Combinatorial, Differential, Sequential, Theory};

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
    /// Corresponds to temporary, local wires.
    temp: Interface<S>,
    /// Corresponds to the initial condition.
    init: Block<I>,
    /// Corresponds to the update action.
    update: Block<J>,
    /// Corresponds to the delay activity.
    delay: Block<F>,
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
        self.temp.wires()
    }

    pub fn empty() -> Self {
        Self {
            ctrl: Interface::empty(),
            wait: Interface::empty(),
            read: Interface::empty(),
            temp: Interface::empty(),
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
        temp: Interface<S>,
        init: Block<I>,
        update: Block<J>,
        delay: Block<F>,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            //================================================================================
            // Check declared wires
            //================================================================================
            let mut decl: HashMap<usize, &S> = HashMap::new();
            // declare read and await, don't allow repetition
            {
                for wire in read.wires().chain(wait.wires()) {
                    debug_assert!(
                        decl.insert(wire.id(), wire.dtype()).is_none(),
                        "wire {} doubly declared",
                        wire.id()
                    );
                }
            }
            // check that read and wait are read only
            {
                let init = init.iter().flat_map(|t| t.write().ids());
                let update = update.iter().flat_map(|t| t.write().ids());
                let delay = delay.iter().flat_map(|t| t.write().ids());
                for id in init.chain(update).chain(delay) {
                    debug_assert!(!decl.contains_key(&id), "invalid write on wire {id}");
                }
            }
            // declare ctrl and temp, don't allow repetition
            for (w, dtype) in ctrl.wires().chain(temp.wires()).map(Into::into) {
                debug_assert!(decl.insert(w, dtype).is_none(), "wire {w} doubly declared");
            }
            // check that read wires of terms have consistent dtype
            {
                let init = init.iter().flat_map(|t| t.read().ids());
                let update = update.iter().flat_map(|t| t.read().ids());
                let delay = delay.iter().flat_map(|t| t.write().ids());
                for id in init.chain(update).chain(delay) {
                    debug_assert!(decl.contains_key(&id), "wire {id} undeclared");
                }
            }
            // check that write wires of terms have consistent dtype
            {
                let init = init.iter().flat_map(|t| t.write().ids());
                let update = update.iter().flat_map(|t| t.write().ids());
                let delay = delay.iter().flat_map(|t| t.write().ids());
                for id in init.chain(update).chain(delay) {
                    debug_assert!(decl.contains_key(&id), "wire {id} undeclared");
                }
            }

            //================================================================================
            // Check init terms
            //================================================================================
            // the init terms can initially read from the await wires of the atom
            let mut written = HashSet::<usize>::from_iter(wait.ids());
            for term in init.iter() {
                // all read wires were written before in the block
                debug_assert!(
                    term.read().ids().all(|rd| written.contains(&rd)),
                    "read before write"
                );
                // no write wire was written before in the block
                debug_assert!(
                    term.write().ids().all(|rd| !written.contains(&rd)),
                    "write after write"
                );
                written.extend(term.write().ids());
            }
            // all control wires are written
            debug_assert!(ctrl.ids().all(|w| written.contains(&w)));

            //================================================================================
            // Check update terms
            //================================================================================
            // the update block can initially read from the read and await wires of the atom
            let mut written = HashSet::<usize>::from_iter(read.ids().chain(wait.ids()));
            for term in update.iter() {
                // all read wires were written before in the block
                debug_assert!(
                    term.read().ids().all(|rd| written.contains(&rd)),
                    "read before write"
                );
                // no write wire was written before in the block
                debug_assert!(
                    term.write().ids().all(|rd| !written.contains(&rd)),
                    "write after write"
                );
                written.extend(term.write().ids());
            }
            // all control wires are written
            debug_assert!(ctrl.ids().all(|w| written.contains(&w)));

            //================================================================================
            // Check delay terms
            //================================================================================
            // the delay block can initially read from the read and await wires of the atom
            let mut written = HashSet::<usize>::from_iter(read.ids().chain(wait.ids()));
            for term in delay.iter() {
                // all read wires were written before in the block
                debug_assert!(
                    term.read().ids().all(|rd| written.contains(&rd)),
                    "read before write"
                );
                // no write wire was written before in the block
                debug_assert!(
                    term.write().ids().all(|rd| !written.contains(&rd)),
                    "write after write"
                );
                written.extend(term.write().ids());
            }
            // all control wires are written
            debug_assert!(ctrl.ids().all(|w| written.contains(&w)));
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

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Clone + Eq,
{
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
    pub fn sequential<'a, W, V, U>(latched: W, next: W, init: V, update: U) -> Result<Self, String>
    where
        W: IntoIterator<Item = &'a Wire<S>>,
        V: IntoIterator<Item = Term<I>>,
        U: IntoIterator<Item = Term<J>>,
        S: 'a + fmt::Display,
    {
        let latched: HashMap<usize, &S> = latched.into_iter().map(Into::into).collect();
        let next: HashMap<usize, &S> = next.into_iter().map(Into::into).collect();

        let init = Block::try_from_iter(init)?;
        let update = Block::try_from_iter(update)?;

        let mut ctrl: BTreeMap<usize, Wire<S>> = BTreeMap::new();
        let mut wait: BTreeMap<usize, Wire<S>> = BTreeMap::new();
        let mut read: BTreeMap<usize, Wire<S>> = BTreeMap::new();
        let mut temp: BTreeMap<usize, Wire<S>> = BTreeMap::new();

        for rd in init.read().iter().map(|[w]| w) {
            // init can only read from await wires
            let next_dtype = next.get(&rd.id());
            if next_dtype.is_some_and(|&d| d == rd.dtype()) {
                wait.insert(rd.id(), rd.clone());
                continue;
            } else if next_dtype.is_some() {
                return Err(format!(
                    "Next wire of wire {} in init has a different dtype",
                    rd.id()
                ));
            }

            if latched.contains_key(&rd.id()) {
                return Err(format!("Init reads latched wire {}", rd.id()));
            }

            // dangling read wires are invalid
            return Err(format!("Wire {} in init is dangling read", rd.id()));
        }

        for rd in update.read().iter().map(|[w]| w) {
            // if the update reads from a next wire, then this is awaited
            // otherwise, this must be read from outside the atom
            let latched_dtype = latched.get(&rd.id());
            if latched_dtype.is_some_and(|&d| d == rd.dtype()) {
                read.insert(rd.id(), rd.clone());
                continue;
            } else if latched_dtype.is_some() {
                return Err(format!("Wire {} in update has wrong dtype", rd.id()));
            }

            let next_dtype = next.get(&rd.id());
            if next_dtype.is_some_and(|&d| d == rd.dtype()) {
                wait.insert(rd.id(), rd.clone());
                continue;
            } else if next_dtype.is_some() {
                return Err(format!(
                    "Next wire of wire {} in update has a different dtype",
                    rd.id()
                ));
            }

            // dangling read wires are parameters
            return Err(format!("Wire {} in update is dangling read", rd.id()));
        }

        for wt in [init.write(), update.write()]
            .into_iter()
            .flatten()
            .map(|[w]| w)
        {
            // if the init/update writes to a next wire, then this wire is controlled
            // otherwise, this wire must be temporary
            let next_dtype = next.get(&wt.id());
            if next_dtype.is_some_and(|&d| d == wt.dtype()) {
                ctrl.insert(wt.id(), wt.clone());
                continue;
            } else if next_dtype.is_some() {
                return Err(format!("Controlled wire {} has a wrong dtype", wt.id()));
            }

            if latched.contains_key(&wt.id()) {
                return Err(format!("Writing a latched wire {}", wt.id()));
            } else {
                temp.insert(wt.id(), wt.clone());
            }
        }

        for &ctr in ctrl.keys() {
            if !init.write().ids().any(|wrt| wrt == ctr) {
                return Err(format!("Controlled wire {} is not written in init", ctr));
            }
            if !update.write().ids().any(|wrt| wrt == ctr) {
                return Err(format!("Controlled wire {} is not written in update", ctr));
            }
        }

        let delay = Block::zero(ctrl.clone().into_values())?;

        Ok(Self::new_unchecked(
            Interface::from_wires_unchecked(ctrl.into_values()),
            Interface::from_wires_unchecked(wait.into_values()),
            Interface::from_wires_unchecked(read.into_values()),
            Interface::from_wires_unchecked(temp.into_values()),
            init,
            update,
            delay,
        ))
    }

    pub fn differential<'a, O, P, Q, R>(wires: O, init: Q, delay: R) -> Result<Self, String>
    where
        P: Into<[&'a Wire<S>; 2]>,
        O: IntoIterator<Item = P>,
        Q: IntoIterator<Item = Term<I>>,
        R: IntoIterator<Item = Term<F>>,
        S: 'a,
    {
        let mut latched = HashSet::<Wire<S>>::new();
        let mut derived = HashMap::<Wire<S>, Wire<S>>::new();

        for [l, d] in wires.into_iter().map(Into::into) {
            if l.dtype() != d.dtype() {
                return Err("dtype mismatch".to_string());
            }
            if latched.contains(l) || derived.contains_key(l) {
                return Err(format!("duplicate wire {}", l.id()));
            }
            if latched.contains(d) || derived.contains_key(d) {
                return Err(format!("duplicate wire {}", d.id()));
            }
            latched.insert(l.clone());
            derived.insert(d.clone(), l.clone());
        }

        let init = Block::try_from_iter(init)?;
        let delay = Block::try_from_iter(delay)?;

        let mut ctrl: BTreeMap<usize, Wire<S>> = BTreeMap::new();
        let mut wait: BTreeMap<usize, Wire<S>> = BTreeMap::new();
        let mut read: BTreeMap<usize, Wire<S>> = BTreeMap::new();
        let mut temp: BTreeMap<usize, Wire<S>> = BTreeMap::new();

        for rd in init.read().iter().map(|[w]| w) {
            // init can only read from await wires
            if derived.contains_key(rd) {
                wait.insert(rd.id(), rd.clone());
                continue;
            }

            if latched.contains(rd) {
                return Err(format!("Init reads latched wire {}", rd.id()));
            }

            // dangling read wires are invalid
            return Err(format!("Wire {} in init is dangling read", rd.id()));
        }

        for rd in delay.read().iter().map(|[w]| w) {
            // if the update reads from a next wire, then this is awaited
            // otherwise, this must be read from outside the atom
            if latched.contains(rd) {
                read.insert(rd.id(), rd.clone());
                continue;
            }

            if derived.contains_key(rd) {
                wait.insert(rd.id(), rd.clone());
                continue;
            }

            // dangling read wires are parameters
            return Err(format!("Wire {} in update is dangling read", rd.id()));
        }

        for wt in [init.write(), delay.write()]
            .into_iter()
            .flatten()
            .map(|[w]| w)
        {
            // if the init/update writes to a next wire, then this wire is controlled
            // otherwise, this wire must be temporary
            if derived.contains_key(wt) {
                ctrl.insert(wt.id(), wt.clone());
                continue;
            }

            if latched.contains(wt) {
                return Err(format!("Controlling a latched wire {}", wt.id()));
            } else {
                temp.insert(wt.id(), wt.clone());
            }
        }

        for &ctr in ctrl.keys() {
            if !init.write().ids().any(|wrt| wrt == ctr) {
                return Err(format!("Controlled wire {} is not written in init", ctr));
            }
            if !delay.write().ids().any(|wrt| wrt == ctr) {
                return Err(format!(
                    "Controlled wire {} is not controlled in delay",
                    ctr
                ));
            }
        }

        let past: Vec<Wire<S>> = ctrl
            .values()
            .map(|w| derived.get(w).unwrap().clone())
            .collect();
        let update = Block::skip(ctrl.values().cloned(), past.iter().cloned())?;

        read.extend(past.into_iter().map(|w| (w.id(), w)));

        Ok(Self::new_unchecked(
            Interface::from_wires_unchecked(ctrl.into_values()),
            Interface::from_wires_unchecked(wait.into_values()),
            Interface::from_wires_unchecked(read.into_values()),
            Interface::from_wires_unchecked(temp.into_values()),
            init,
            update,
            delay,
        ))
    }
}

impl<I, J, F, S> Atom<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Eq + Clone + Debug,
{
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
    pub fn combinatorial<'a, T, N, V>(next: N, assign: V) -> Result<Self, String>
    where
        T: Theory<Sort = S> + Into<I> + Into<J> + Clone,
        V: IntoIterator<Item = Term<T>>,
        N: IntoIterator<Item = &'a Wire<S>>,
        S: 'a,
    {
        let next: HashMap<usize, &S> = next.into_iter().map(Into::into).collect();
        let assign: Block<T> = Block::try_from_iter(assign)?;

        let mut ctrl: BTreeMap<usize, Wire<S>> = BTreeMap::new();
        let mut wait: BTreeMap<usize, Wire<S>> = BTreeMap::new();
        let mut temp: BTreeMap<usize, Wire<S>> = BTreeMap::new();

        for rd in assign.read().iter().map(|[w]| w) {
            //  can only read from await wires
            let expected_dtype = next.get(&rd.id());
            if expected_dtype.is_some_and(|&d| d == rd.dtype()) {
                wait.insert(rd.id(), rd.clone());
            } else if expected_dtype.is_some() {
                return Err(format!(
                    "Read wire {} from `assign` has a different dtype than its next version",
                    rd.id()
                ));
            } else {
                return Err(format!("Read wire {} in assign", rd.id()));
            }
        }

        for wt in assign.write().iter().map(|[w]| w) {
            // if it writes to a next wire, then this wire is controlled
            // otherwise, this wire must be temporary
            let expected_dtype = next.get(&wt.id());
            if expected_dtype.is_some_and(|&d| d == wt.dtype()) {
                ctrl.insert(wt.id(), wt.clone());
            } else if expected_dtype.is_some() {
                return Err(format!(
                    "Write wire {} from `assign` has a different dtype than its next version",
                    wt.id()
                ));
            } else {
                temp.insert(wt.id(), wt.clone());
            }
        }

        let init: Block<I> = Block::try_from_iter(assign.iter().cloned())?;
        let update: Block<J> = Block::try_from_iter(assign)?;
        let delay = Block::zero(ctrl.clone().into_values())?;

        Ok(Self::new_unchecked(
            Interface::from_wires_unchecked(ctrl.into_values()),
            Interface::from_wires_unchecked(wait.into_values()),
            Interface::empty(),
            Interface::from_wires_unchecked(temp.into_values()),
            init,
            update,
            delay,
        ))
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
        for (i, wr) in self.ctrl.ids().enumerate() {
            if i == 0 {
                write!(f, " {BOLD}controls{RESET} w{wr}")?;
            } else {
                write!(f, ", w{wr}")?;
            }
        }
        for (i, wr) in self.read.ids().enumerate() {
            if i == 0 {
                write!(f, " {BOLD}reads{RESET} w{wr}")?;
            } else {
                write!(f, ", w{wr}")?;
            }
        }
        for (i, wr) in self.wait.ids().enumerate() {
            if i == 0 {
                write!(f, " {BOLD}awaits{RESET} w{wr}")?;
            } else {
                write!(f, ", w{wr}")?;
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
    S: fmt::Display + Debug + Clone,
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
    pub fn sequential<O, P, Q, R>(obs: O, init: Q, update: R) -> Result<Self, String>
    where
        P: Into<[Wire<S>; 2]>,
        O: IntoIterator<Item = P>,
        Q: IntoIterator<Item = Term<I>>,
        R: IntoIterator<Item = Term<J>>,
    {
        Self::partially_observable_sequential(obs, std::iter::empty::<P>(), init, update)
    }

    pub fn partially_observable_sequential<TO, TP, O, P, Q, R>(
        obs: O,
        prvt: P,
        init: Q,
        update: R,
    ) -> Result<Self, String>
    where
        TO: Into<[Wire<S>; 2]>,
        TP: Into<[Wire<S>; 2]>,
        O: IntoIterator<Item = TO>,
        P: IntoIterator<Item = TP>,
        Q: IntoIterator<Item = Term<I>>,
        R: IntoIterator<Item = Term<J>>,
    {
        let obs = Interface::try_from_iter(obs)?;
        let prvt = Interface::try_from_iter(prvt)?;
        let latched = obs.latched().iter().chain(prvt.latched().iter());
        let next = obs.next().iter().chain(prvt.next().iter());
        let atom = Atom::sequential(latched, next, init, update)?;
        Self::partially_observable(obs, prvt, std::iter::once(atom))
    }

    pub fn differential<O, P, Q, R>(obs: O, init: Q, delay: R) -> Result<Self, String>
    where
        P: Into<[Wire<S>; 2]>,
        O: IntoIterator<Item = P>,
        Q: IntoIterator<Item = Term<I>>,
        R: IntoIterator<Item = Term<F>>,
    {
        let obs = obs.into_iter().map(Into::into).collect::<Vec<_>>();
        let atom = Atom::differential(obs.iter().map(|[a, b]| [a, b]), init, delay)?;
        Self::observable(obs, std::iter::once(atom))
    }
}

impl<I, J, F, S> Module<I, J, F, S>
where
    I: Combinatorial<Sort = S>,
    J: Sequential<Sort = S>,
    F: Differential<Sort = S>,
    S: Eq + Clone + Debug + fmt::Display,
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
    /// - [`Module::partially_observable_sequential`], for constructing stateful, sequential modules.
    /// - [`Atom::combinatorial`], for creating individual combinatorial atoms.
    pub fn combinatorial<T, R, O, V>(obs: O, assign: V) -> Result<Self, String>
    where
        T: Theory<Sort = S> + Into<I> + Into<J> + Clone,
        R: Into<[Wire<T::Sort>; 2]>,
        O: IntoIterator<Item = R>,
        V: IntoIterator<Item = Term<T>>,
    {
        let obs = Interface::from_iter(obs);
        let atom = Atom::combinatorial(obs.next(), assign)?;
        Self::observable(obs, [atom])
    }
}
