use crate::term::{Block, Term};
use crate::wire::{Interface, Wire};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use theory::Theory;

#[cfg(debug_assertions)]
use std::collections::HashSet;

/// This data structure corresponds to the atom of reactive modules.
#[derive(Debug, Clone)]
pub struct Atom<T: Theory> {
    /// Corresponds to ctr variables.
    ctrl: Interface<T::DType>,
    /// Corresponds to wait variables.
    wait: Interface<T::DType>,
    /// Corresponds to read variables.
    read: Interface<T::DType>,
    /// Corresponds to temporary, local wires.
    temp: Interface<T::DType>,
    /// Corresponds to the initial action.
    init: Block<T>,
    /// Corresponds to the update action.
    update: Block<T>,
    // flow: Vec<Term<I>>, // the default flow must be a constant so the derivative is 0
}

impl<T: Theory> Atom<T> {
    /// Returns a reference to the initial action.
    pub fn init(&self) -> &Block<T> {
        &self.init
    }
    /// Returns a reference to the update action.
    pub fn update(&self) -> &Block<T> {
        &self.update
    }

    // fn flow(&self) -> &[Term<I>] {
    //     &self.flow
    // }

    pub fn ctrl(&self) -> &Interface<T::DType> {
        &self.ctrl
    }

    pub fn wait(&self) -> &Interface<T::DType> {
        &self.wait
    }

    pub fn read(&self) -> &Interface<T::DType> {
        &self.read
    }

    pub fn temp(&self) -> impl Iterator<Item = &Wire<T::DType>> {
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

impl<T: Theory> Default for Atom<T> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<T: Theory> Atom<T>
where
    T::DType: Eq,
{
    /// Returns true if this atoms awaits the other atom
    pub fn awaits(&self, other: &Atom<T>) -> bool {
        !self.wait.is_disjoint(&other.ctrl)
    }

    /// Creates an atom from its components. This method checks the inputs only using assertions
    /// in debug mode.
    fn new_unchecked(
        ctrl: Interface<T::DType>,
        wait: Interface<T::DType>,
        read: Interface<T::DType>,
        temp: Interface<T::DType>,
        init: Block<T>,
        update: Block<T>,
    ) -> Self {
        #[cfg(debug_assertions)]
        {
            //================================================================================
            // Check declared wires
            //================================================================================
            let mut decl: HashMap<usize, &T::DType> = HashMap::new();
            // declare read and await, don't allow repetition
            {
                let read_wait_param = read.wires().chain(wait.wires());
                for (w, dtype) in read_wait_param.map(Into::into) {
                    debug_assert!(decl.insert(w, dtype).is_none(), "wire {w} doubly declared");
                }
            }
            // check that read and wait are read only
            {
                let init_update = init.iter().chain(update.iter());
                for id in init_update.flat_map(|t| t.write().wires()).map(Wire::id) {
                    debug_assert!(!decl.contains_key(&id), "wire {id} undeclared");
                }
            }
            // declare ctrl and temp, don't allow repetition
            for (w, dtype) in ctrl.wires().chain(temp.wires()).map(Into::into) {
                debug_assert!(decl.insert(w, dtype).is_none(), "wire {w} doubly declared");
            }
            // check that read wires of terms have consistent dtype
            {
                let init_update = init.iter().chain(update.iter());
                for (w, dtype) in init_update.flat_map(|t| t.read().wires()).map(Into::into) {
                    debug_assert!(
                        decl.insert(w, dtype).is_some_and(|d| d == dtype),
                        "wire {w} undeclared or dtype mismatch"
                    );
                }
            }
            // check that write wires of terms have consistent dtype
            {
                let init_update = init.iter().chain(update.iter());
                for (w, dtype) in init_update.flat_map(|t| t.write().wires()).map(Into::into) {
                    debug_assert!(
                        decl.insert(w, dtype).is_some_and(|d| d == dtype),
                        "wire {w} undeclared or dtype mismatch"
                    );
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

impl<T: Theory> Atom<T>
where
    T::DType: Eq + Clone,
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
    /// - [`Module::sequential`], for creating sequential modules.
    pub fn sequential<'a, L, N, V, U>(
        latched: L,
        next: N,
        init: V,
        update: U,
    ) -> Result<Self, &'static str>
    where
        L: IntoIterator<Item = &'a Wire<T::DType>>,
        N: IntoIterator<Item = &'a Wire<T::DType>>,
        V: IntoIterator<Item = Term<T>>,
        U: IntoIterator<Item = Term<T>>,
        T::DType: 'a,
    {
        let latched: HashMap<usize, &T::DType> = latched.into_iter().map(Into::into).collect();
        let next: HashMap<usize, &T::DType> = next.into_iter().map(Into::into).collect();

        let init = Block::try_from_iter(init)?;
        let update = Block::try_from_iter(update)?;

        let mut ctrl: BTreeMap<usize, Wire<T::DType>> = BTreeMap::new();
        let mut wait: BTreeMap<usize, Wire<T::DType>> = BTreeMap::new();
        let mut read: BTreeMap<usize, Wire<T::DType>> = BTreeMap::new();
        let mut temp: BTreeMap<usize, Wire<T::DType>> = BTreeMap::new();

        for rd in init.read().iter().map(|[w]| w) {
            // init can only read from await wires
            let next_dtype = next.get(&rd.id());
            if next_dtype.is_some_and(|&d| d == rd.dtype()) {
                wait.insert(rd.id(), rd.clone());
                continue;
            } else if next_dtype.is_some() {
                return Err("dtype mismatch");
            }

            if latched.contains_key(&rd.id()) {
                return Err("Init reads latched wire");
            }

            // dangling read wires are invalid
            return Err("Invalid read wire");
        }

        for rd in update.read().iter().map(|[w]| w) {
            // if the update reads from a next wire, then this is awaited
            // otherwise, this must be read from outside the atom
            let latched_dtype = latched.get(&rd.id());
            if latched_dtype.is_some_and(|&d| d == rd.dtype()) {
                read.insert(rd.id(), rd.clone());
                continue;
            } else if latched_dtype.is_some() {
                return Err("dtype mismatch");
            }

            let next_dtype = next.get(&rd.id());
            if next_dtype.is_some_and(|&d| d == rd.dtype()) {
                wait.insert(rd.id(), rd.clone());
                continue;
            } else if next_dtype.is_some() {
                return Err("dtype mismatch");
            }

            // dangling read wires are parameters
            return Err("Invalid read wire");
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
                return Err("dtype mismatch");
            }

            if latched.contains_key(&wt.id()) {
                return Err("write on latched");
            } else {
                temp.insert(wt.id(), wt.clone());
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
            Interface::from_wires_unchecked(ctrl.into_values()),
            Interface::from_wires_unchecked(wait.into_values()),
            Interface::from_wires_unchecked(read.into_values()),
            Interface::from_wires_unchecked(temp.into_values()),
            init,
            update,
        ))
    }
}

impl<T: Theory> Atom<T>
where
    T: Clone,
    T::DType: Eq + Clone,
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
    pub fn combinatorial<'a, N, V>(next: N, assign: V) -> Result<Self, &'static str>
    where
        V: IntoIterator<Item = Term<T>>,
        N: IntoIterator<Item = &'a Wire<T::DType>>,
        T::DType: 'a,
    {
        let next: HashMap<usize, &T::DType> = next.into_iter().map(Into::into).collect();
        let assign = Block::try_from_iter(assign)?;

        let mut ctrl: BTreeMap<usize, Wire<T::DType>> = BTreeMap::new();
        let mut wait: BTreeMap<usize, Wire<T::DType>> = BTreeMap::new();
        let mut temp: BTreeMap<usize, Wire<T::DType>> = BTreeMap::new();
        let mut param: BTreeMap<usize, Wire<T::DType>> = BTreeMap::new();

        for rd in assign.read().iter().map(|[w]| w) {
            //  can only read from await wires
            let expected_dtype = next.get(&rd.id());
            if expected_dtype.is_some_and(|&d| d == rd.dtype()) {
                wait.insert(rd.id(), rd.clone());
            } else if expected_dtype.is_some() {
                return Err("dtype mismatch");
            } else {
                param.insert(rd.id(), rd.clone());
            }
        }

        for wt in assign.write().iter().map(|[w]| w) {
            // if it writes to a next wire, then this wire is controlled
            // otherwise, this wire must be temporary
            let expected_dtype = next.get(&wt.id());
            if expected_dtype.is_some_and(|&d| d == wt.dtype()) {
                ctrl.insert(wt.id(), wt.clone());
            } else if expected_dtype.is_some() {
                return Err("dtype mismatch");
            } else {
                temp.insert(wt.id(), wt.clone());
            }
        }

        Ok(Self::new_unchecked(
            Interface::from_wires_unchecked(ctrl.into_values()),
            Interface::from_wires_unchecked(wait.into_values()),
            Interface::empty(),
            Interface::from_wires_unchecked(temp.into_values()),
            assign.clone(),
            assign,
        ))
    }
}

impl<T: Theory> Atom<T>
where
    T: fmt::Display,
    T::DType: fmt::Display,
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
        writeln!(f, "{pad}{BOLD}update{RESET}")?;
        for term in self.update.iter() {
            writeln!(f, "{pad}{INDENT}{term}")?;
        }
        Ok(())
    }
}

impl<T: Theory> fmt::Display for Atom<T>
where
    T: fmt::Display,
    T::DType: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_indent(f, "")
    }
}
