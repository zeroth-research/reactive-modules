use crate::term::{Block, Term};
use crate::wire::{Interface, Wire};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

/// This data structure corresponds to the atom of reactive modules.
#[derive(Debug, Clone)]
pub struct Atom<D, I> {
    /// Corresponds to ctr variables.
    ctrl: Interface<D>,
    /// Corresponds to wait variables.
    wait: Interface<D>,
    /// Corresponds to read variables.
    read: Interface<D>,
    /// Corresponds to temporary, local wires.
    temp: Interface<D>,

    /// Corresponds to the initial action.
    init: Block<D, I>,
    /// Corresponds to the update action.
    update: Block<D, I>,
    // flow: Vec<Term<I>>, // the default flow must be a constant so the derivative is 0
}
impl<D, I> Atom<D, I> {
    /// Returns a reference to the initial action.
    pub fn init(&self) -> &Block<D, I> {
        &self.init
    }
    /// Returns a reference to the update action.
    pub fn update(&self) -> &Block<D, I> {
        &self.update
    }

    // fn flow(&self) -> &[Term<I>] {
    //     &self.flow
    // }

    pub fn ctrl(&self) -> &Interface<D> {
        &self.ctrl
    }

    pub fn wait(&self) -> &Interface<D> {
        &self.wait
    }

    pub fn read(&self) -> &Interface<D> {
        &self.read
    }

    pub fn temp(&self) -> impl Iterator<Item = &Wire<D>> {
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
        }
    }
}

impl<D, I> Default for Atom<D, I> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<D: Eq, I> Atom<D, I> {
    /// Returns true if this atoms awaits the other atom
    pub fn awaits(&self, other: &Atom<D, I>) -> bool {
        !self.wait.is_disjoint(&other.ctrl)
    }

    /// Creates an atom from its components. This method checks the inputs only using assertions
    /// in debug mode.
    fn new_unchecked(
        ctrl: Interface<D>,
        wait: Interface<D>,
        read: Interface<D>,
        temp: Interface<D>,
        init: Block<D, I>,
        update: Block<D, I>,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            //================================================================================
            // Check declared wires
            //================================================================================
            let mut decl: HashMap<usize, &D> = HashMap::new();
            // declare read and await, don't allow repetition
            for (w, dtype) in read.wires().chain(wait.wires()).map(Into::into) {
                debug_assert!(decl.insert(w, dtype).is_none(), "wire {w} doubly declared");
            }
            // check that read and wait are read only
            for id in init
                .iter()
                .chain(update.iter())
                .flat_map(|t| t.write()[0].iter())
                .map(Wire::id)
            {
                debug_assert!(!decl.contains_key(&id), "wire {id} undeclared");
            }
            // declare ctrl and temp, don't allow repetition
            for (w, dtype) in ctrl.wires().chain(temp.wires()).map(Into::into) {
                debug_assert!(decl.insert(w, dtype).is_none(), "wire {w} doubly declared");
            }
            // check that read wires of terms have consistent dtype
            for (w, dtype) in init
                .iter()
                .chain(update.iter())
                .flat_map(|t| t.read().wires())
                .map(Into::into)
            {
                debug_assert!(
                    decl.insert(w, dtype).is_some_and(|d| d == dtype),
                    "wire {w} undeclared or dtype mismatch"
                );
            }
            // check that write wires of terms have consistent dtype
            for (w, dtype) in init
                .iter()
                .chain(update.iter())
                .flat_map(|t| t.write().wires())
                .map(Into::into)
            {
                debug_assert!(
                    decl.insert(w, dtype).is_some_and(|d| d == dtype),
                    "wire {w} undeclared or dtype mismatch"
                );
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
        }

        Self {
            ctrl,
            wait,
            read,
            temp,
            init,
            update,
        }
    }
}

impl<D: Eq + Clone, I> Atom<D, I> {
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
    /// - [`Module::sequential`], for creating sequential modules.
    pub fn sequential<'a, L, N, V, U>(
        latched: L,
        next: N,
        init: V,
        update: U,
    ) -> Result<Self, &'static str>
    where
        L: IntoIterator<Item = &'a Wire<D>>,
        N: IntoIterator<Item = &'a Wire<D>>,
        V: IntoIterator<Item = Term<D, I>>,
        U: IntoIterator<Item = Term<D, I>>,
        D: 'a,
    {
        let latched: HashMap<usize, &D> = latched.into_iter().map(Into::into).collect();
        let next: HashMap<usize, &D> = next.into_iter().map(Into::into).collect();

        let init = Block::try_from_iter(init)?;
        let update = Block::try_from_iter(update)?;

        let mut ctrl: BTreeMap<usize, D> = BTreeMap::new();
        let mut wait: BTreeMap<usize, D> = BTreeMap::new();
        let mut read: BTreeMap<usize, D> = BTreeMap::new();
        let mut temp: BTreeMap<usize, D> = BTreeMap::new();

        for (rd, dtype) in init.read().iter().map(|[w]| w.into()) {
            // init can only read from await wires
            let next_dtype = next.get(&rd);
            if next_dtype.is_some_and(|&d| d == dtype) {
                wait.insert(rd, dtype.clone());
                continue;
            } else if next_dtype.is_some() {
                return Err("dtype mismatch");
            }

            if latched.contains_key(&rd) {
                return Err("Init reads latched wire");
            } else {
                return Err("invalid wire");
            }
        }

        for (rd, dtype) in update.read().iter().map(|[w]| w.into()) {
            // if the update reads from a next wire, then this is awaited
            // otherwise, this must be read from outside the atom
            let latched_dtype = latched.get(&rd);
            if latched_dtype.is_some_and(|&d| d == dtype) {
                read.insert(rd, dtype.clone());
                continue;
            } else if latched_dtype.is_some() {
                return Err("dtype mismatch");
            }

            let next_dtype = next.get(&rd);
            if next_dtype.is_some_and(|&d| d == dtype) {
                wait.insert(rd, dtype.clone());
                continue;
            } else if next_dtype.is_some() {
                return Err("dtype mismatch");
            }

            return Err("invalid wire");
        }

        for (wt, dtype) in [init.write(), update.write()]
            .into_iter()
            .flatten()
            .map(|[w]| w.into())
        {
            // if the init/update writes to a next wire, then this wire is controlled
            // otherwise, this wire must be temporary
            let next_dtype = next.get(&wt);
            if next_dtype.is_some_and(|&d| d == dtype) {
                ctrl.insert(wt, dtype.clone());
                continue;
            } else if next_dtype.is_some() {
                return Err("dtype mismatch");
            }

            if latched.contains_key(&wt) {
                return Err("invalid write");
            } else {
                temp.insert(wt, dtype.clone());
            }
        }

        for (&ctr, _) in ctrl.iter() {
            if !init.write().ids().any(|wrt| wrt == ctr) {
                return Err("unassigned control wire after init");
            }
            if !update.write().ids().any(|wrt| wrt == ctr) {
                return Err("unassigned control wire after update");
            }
        }

        Ok(Self::new_unchecked(
            Interface::from_wires_unchecked(ctrl),
            Interface::from_wires_unchecked(wait),
            Interface::from_wires_unchecked(read),
            Interface::from_wires_unchecked(temp),
            init,
            update,
        ))
    }
}

impl<D: Eq + Clone, I: Clone> Atom<D, I> {
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
    pub fn combinatorial<'a, N, V>(next: N, assign: V) -> Result<Self, &'static str>
    where
        V: IntoIterator<Item = Term<D, I>>,
        N: IntoIterator<Item = &'a Wire<D>>,
        D: 'a,
    {
        let next: HashMap<usize, &D> = next.into_iter().map(Into::into).collect();
        let assign = Block::try_from_iter(assign)?;

        let mut ctrl: BTreeMap<usize, D> = BTreeMap::new();
        let mut wait: BTreeMap<usize, D> = BTreeMap::new();
        let mut temp: BTreeMap<usize, D> = BTreeMap::new();

        for (rd, dtype) in assign.read().iter().map(|[w]| w.into()) {
            //  can only read from await wires
            let expected_dtype = next.get(&rd);
            if expected_dtype.is_some_and(|&d| d == dtype) {
                wait.insert(rd, dtype.clone());
            } else if expected_dtype.is_some() {
                return Err("dtype mismatch");
            } else {
                return Err("invalid wire");
            }
        }

        for (wt, dtype) in assign.write().iter().map(|[w]| w.into()) {
            // if it writes to a next wire, then this wire is controlled
            // otherwise, this wire must be temporary
            let expected_dtype = next.get(&wt);
            if expected_dtype.is_some_and(|&d| d == dtype) {
                ctrl.insert(wt, dtype.clone());
            } else if expected_dtype.is_some() {
                return Err("dtype mismatch");
            } else {
                temp.insert(wt, dtype.clone());
            }
        }

        Ok(Self::new_unchecked(
            Interface::from_wires_unchecked(ctrl),
            Interface::from_wires_unchecked(wait),
            Interface::empty(),
            Interface::from_wires_unchecked(temp),
            assign.clone(),
            assign,
        ))
    }
}

impl<D: fmt::Display, I: fmt::Display> Atom<D, I> {
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
        writeln!(f, "{pad}{BOLD}update{RESET}")?;
        for term in self.update.iter() {
            writeln!(f, "{pad}{INDENT}{term}")?;
        }
        Ok(())
    }
}

impl<D: fmt::Display, I: fmt::Display> fmt::Display for Atom<D, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, "")
    }
}
