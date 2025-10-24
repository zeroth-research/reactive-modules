use crate::term::{Instruction, Term};
use crate::wire::Wire;

/// This data structure corresponds to the atom of reactive modules.
pub struct Atom<D, I: Instruction> {
    /// Corresponds to read variables.
    read: Wire<D>,
    /// Corresponds to ctr variables.
    write: Wire<D>,
    /// Corresponds to wait variables.
    wait: Wire<D>,

    /// Corresponds to the initial action.
    init: Vec<Term<D, I>>,
    /// Corresponds to the update action.
    update: Vec<Term<D, I>>,
    // delay: Vec<Term<I>>, // the default delay must be a constant so the derivative is 0
}
impl<D, I: Instruction> Atom<D, I> {
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
        &self.write
    }
    pub fn waits(&self) -> &Wire<D> {
        &self.wait
    }

    // fn delay(&self) -> &[Term<I>] {
    //     &self.delay
    // }

    /// Creates an atom from its components. This method checks the inputs only using assertions
    /// in debug mode.
    pub fn new_unchecked(
        read: Wire<D>,
        write: Wire<D>,
        wait: Wire<D>,
        init: Vec<Term<D, I>>,
        update: Vec<Term<D, I>>,
    ) -> Self {
        Self {
            read,
            write,
            wait,
            init,
            update,
        }
    }
}
