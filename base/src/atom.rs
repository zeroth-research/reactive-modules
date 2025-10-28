use crate::term::Term;
use crate::wire::Wire;

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
    fn init(&self) -> &[Term<D, I>] {
        &self.init
    }
    /// Returns a reference to the update action.
    fn update(&self) -> &[Term<D, I>] {
        &self.update
    }

    // fn delay(&self) -> &[Term<I>] {
    //     &self.delay
    // }

    /// Creates an atom from its components. This method checks the inputs only using assertions
    /// in debug mode.
    pub fn new_unchecked(
        ctrl: Wire<D>,
        wait: Wire<D>,
        read: Wire<D>,
        init: Vec<Term<D, I>>,
        update: Vec<Term<D, I>>,
    ) -> Self {
        Self {
            ctrl,
            wait,
            read,
            init,
            update,
        }
    }
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
}

impl<D: Eq + Clone, I> Atom<D, I> {
    pub fn with_module_wire(
        wire: &[Wire<D>; 2],
        init: Vec<Term<D, I>>,
        update: Vec<Term<D, I>>,
    ) -> Result<Self, &'static str> {
        // Check latched and next wires
        if !wire[0].is_twin(&wire[1]) {
            return Err("latched and next wires are not matching");
        }

        let mut ctrl = Wire::none();
        let mut wait = Wire::none();
        let mut read = Wire::none();

        let mut written = Wire::none();

        for term in init.iter() {
            if !term.read.is_disjoint(&wire[0]) {
                return Err("init read from latched");
            }

            if !term.read.is_disjoint(&wire[1]) {
                wait = wait
                    .union(&term.read.intersection(&wire[1]).unwrap())
                    .unwrap();
                written = written.union(&wait).unwrap();
                //TODO implement mutating union (insert) and remove the guard
            }

            if term.read.is_disjoint(&written) {
                return Err("read before write");
            }
            if !term.write.is_disjoint(&written) {
                return Err("write after write");
            }

            if !term.write.is_disjoint(&wire[1]) {
                ctrl = ctrl
                    .union(&term.write.intersection(&wire[1]).unwrap())
                    .unwrap();
                //TODO implement mutating union (insert) and remove the guard
            }
            written = written.union(&term.write).unwrap();
        }

        written = wire[0].union(&wait).unwrap();

        for term in update.iter() {
            if !term.read.is_disjoint(&wire[0]) {
                read = read
                    .union(&term.read.intersection(&wire[0]).unwrap())
                    .unwrap();
                //TODO implement mutating union (insert) and remove the guard
            }
            if !term.read.is_disjoint(&wire[1]) {
                wait = wait
                    .union(&term.read.intersection(&wire[1]).unwrap())
                    .unwrap();
                written = written.union(&wait).unwrap();
                //TODO implement mutating union (insert) and remove the guard
            }
            if term.read.is_disjoint(&written) {
                return Err("read before write");
            }
            if !term.write.is_disjoint(&written) {
                return Err("write after write");
            }
            if !term.write.is_disjoint(&wire[1]) {
                ctrl = ctrl
                    .union(&term.write.intersection(&wire[1]).unwrap())
                    .unwrap();
                //TODO implement mutating union (insert) and remove the guard
            }
            written = written.union(&term.write).unwrap();
        }

        Ok(Self::new_unchecked(ctrl, wait, read, init, update))
    }
}
