use crate::term::Term;
use crate::wire::Wire;
use std::collections::HashMap;
use std::ffi::c_ushort;

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

    pub fn reads(&self) -> &Wire<D> {
        &self.read
    }
    pub fn writes(&self) -> &Wire<D> {
        &self.ctrl
    }
    pub fn waits(&self) -> &Wire<D> {
        &self.wait
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
        // debug_assert!(ctrl.is_disjoint(&wait));
        // debug_assert!(ctrl.is_disjoint(&read));
        // debug_assert!(wait.is_disjoint(&read));

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
        if wire[0].size() != wire[1].size() {
            return Err("len mismatch in latched and next wires");
        }
        let mut latched_to_dtype: HashMap<usize, &D> = HashMap::new();
        let mut next_to_latched: HashMap<usize, usize> = HashMap::new();
        for ((a, at), (b, bt)) in wire[0].iter().zip(wire[1].iter()) {
            if at != bt {
                return Err("dtype mismatch in latched and next wires");
            }
            if latched_to_dtype.insert(a, &at).is_some() {
                return Err("duplicate latched wire");
            }
            if next_to_latched.insert(a, b).is_some() {
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

            for (a, at) in wire[1].iter() {
                // if any term writes a next, then this is controlled by the atom
                match write_iter.clone().find(|&(b, bt)| a == b) {
                    Some((b, bt)) => {
                        if at != bt {
                            return Err("dtype mismatch");
                        }
                        ctrl.push((b, bt.clone()));
                    }
                    None => {}
                }
                // if any term reads a next, then this is awaited by the atom
                match read_iter.clone().find(|&(b, bt)| a == b) {
                    Some((b, bt)) => {
                        if at != bt {
                            return Err("dtype mismatch");
                        }
                        wait.push((b, bt.clone()));
                    }
                    None => {}
                }
            }

            for (a, at) in wire[0].iter() {
                // if any term reads a latched, then this is read by the atom
                match read_iter.clone().find(|&(b, bt)| a == b) {
                    Some((b, bt)) => {
                        if at != bt {
                            return Err("dtype mismatch");
                        }
                        read.push((b, bt.clone()));
                    }
                    None => {}
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
