use crate::term::Term;
use crate::wire::Wire;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fmt::Write;

/// This data structure corresponds to the atom of reactive modules.
#[derive(Debug)]
pub struct Atom<D, I> {
    /// Corresponds to ctr variables.
    pub(crate) ctrl: Wire<D>,
    /// Corresponds to wait variables.
    pub(crate) wait: Wire<D>,
    /// Corresponds to read variables.
    pub(crate) read: Wire<D>,

    /// Corresponds to the initial action.
    init: Vec<Term<D, I>>,
    /// Corresponds to the update action.
    update: Vec<Term<D, I>>,
    // delay: Vec<Term<I>>, // the default delay must be a constant so the derivative is 0
}
impl<D, I> Atom<D, I> {
    /// Returns a reference to the initial action.
    pub fn init(&self) -> &[Term<D, I>] {
        &self.init
    }
    /// Returns a reference to the update action.
    pub fn update(&self) -> &[Term<D, I>] {
        &self.update
    }

    // fn delay(&self) -> &[Term<I>] {
    //     &self.delay
    // }

    pub fn ctrl(&self) -> &Wire<D> {
        &self.ctrl
    }
    pub fn wait(&self) -> &Wire<D> {
        &self.wait
    }
    pub fn read(&self) -> &Wire<D> {
        &self.read
    }
}

impl<D: Eq, I> Atom<D, I> {
    /// Returns true if this atoms awaits the other atom
    pub fn awaits(&self, other: &Atom<D, I>) -> bool {
        self.wait.is_subset(&other.ctrl)
    }

    /// Creates an atom from its components. This method checks the inputs only using assertions
    /// in debug mode.
    pub fn new_unchecked(
        ctrl: Wire<D>,
        wait: Wire<D>,
        read: Wire<D>,
        init: Vec<Term<D, I>>,
        update: Vec<Term<D, I>>,
    ) -> Self {
        debug_assert!(ctrl.is_disjoint(&wait));
        debug_assert!(ctrl.is_disjoint(&read));
        debug_assert!(wait.is_disjoint(&read));
        // TODO full consistency check

        Self {
            ctrl,
            wait,
            read,
            init,
            update,
        }
    }
}

impl<D: Eq + Clone, I> Atom<D, I> {
    pub fn with_module_wire(
        wire: &[Wire<D>; 2],
        init: Vec<Term<D, I>>,
        update: Vec<Term<D, I>>,
    ) -> Result<Self, &'static str> {
        // Check and store wire index + dtype information
        if wire[0].len() != wire[1].len() {
            return Err("len mismatch in latched and next wires");
        }
        let mut ltc_set: HashSet<usize> = HashSet::new();
        let mut nxt_to_ltc: HashMap<usize, usize> = HashMap::new();
        for ((ltc, dtype), (nxt, ntype)) in wire[0].iter().zip(wire[1].iter()) {
            if dtype != ntype {
                return Err("dtype mismatch in latched and next wires");
            }
            if !ltc_set.insert(ltc) {
                return Err("duplicate latched wire");
            }
            if nxt_to_ltc.insert(nxt, ltc).is_some() {
                return Err("duplicate next wire");
            }
        }

        let mut ctrl = Vec::new();
        let mut wait = Vec::new();
        let mut read = Vec::new();

        // TODO checks write after write and read before write. The code below will get simpler
        {
            let terms_iter = init.iter().chain(update.iter());
            let write_iter = terms_iter.clone().flat_map(|t| t.write.iter());
            let read_iter = terms_iter.flat_map(|t| t.read.iter());

            for (nxt, dtype) in &wire[1] {
                // if any term writes a next, then this is controlled by the atom
                if let Some((wr, wtype)) = write_iter.clone().find(|&(wr, _)| nxt == wr) {
                    if dtype != wtype {
                        return Err("dtype mismatch");
                    }
                    ctrl.push((wr, dtype.clone()));
                }
                // if any term reads a next, then this is awaited by the atom
                if let Some((wr, wtype)) = read_iter.clone().find(|&(wr, _)| nxt == wr) {
                    if dtype != wtype {
                        return Err("dtype mismatch");
                    }
                    wait.push((wr, dtype.clone()));
                }
            }

            for (ltc, dtype) in &wire[0] {
                // if any term reads a latched, then this is read by the atom
                if let Some((wr, wtype)) = read_iter.clone().find(|&(wr, _)| ltc == wr) {
                    if dtype != wtype {
                        return Err("dtype mismatch");
                    }
                    read.push((wr, dtype.clone()))
                }
            }
        }

        Ok(Self::new_unchecked(
            Wire::new_unchecked(ctrl),
            Wire::new_unchecked(wait),
            Wire::new_unchecked(read),
            init,
            update,
        ))
    }
}

impl<D: fmt::Display, I: fmt::Display> Atom<D, I> {
    pub(crate) fn fmt_indent(&self, f: &mut fmt::Formatter<'_>, pad: &str) -> fmt::Result {
        const BOLD: &str = "\x1b[1m";
        const RESET: &str = "\x1b[0m";
        const INDENT: &str = "  ";

        write!(f, "{pad}{BOLD}atom{RESET}")?;
        for (i, (wr, _)) in self.ctrl.iter().enumerate() {
            if i == 0 {
                write!(f, " {BOLD}controls{RESET} w{wr}")?;
            } else {
                write!(f, ", w{wr}")?;
            }
        }
        for (i, (wr, _)) in self.read.iter().enumerate() {
            if i == 0 {
                write!(f, " {BOLD}reads{RESET} w{wr}")?;
            } else {
                write!(f, ", w{wr}")?;
            }
        }
        for (i, (wr, _)) in self.wait.iter().enumerate() {
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
