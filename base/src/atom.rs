use crate::term::Term;
use crate::wire::Wire;

/// This data structure corresponds to the atom of reactive modules.
#[derive(Debug)]
pub struct Atom<D, I> {
    /// Corresponds to ctr variables.
    pub(crate) ctrl: Wire<D>,
    /// Corresponds to read variables.
    pub(crate) read: Wire<D>,
    /// Corresponds to wait variables.
    pub(crate) wait: Wire<D>,

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
}

impl<D: Eq, I> Atom<D, I> {
    pub fn awaits(&self, other: &Atom<D, I>) -> bool {
        self.wait.is_subset(&other.ctrl)
    }
}
