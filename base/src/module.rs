use crate::atom::Atom;
use crate::term::Instruction;
use crate::wire::Wire;

/// This data structure corresponds to the module of reactive modules.
struct Module<D, I: Instruction> {
    /// Corresponds to the extl variables.
    /// These [Wire]s have two outer dimensions:
    /// - 0: current/latched variables
    /// - 1: next variables
    external: Wire<D>,
    /// Corresponds to the intf variables.
    /// These [Wire]s have two outer dimensions:
    /// - 0: current/latched variables
    /// - 1: next variables
    interface: Wire<D>,
    /// Corresponds to the priv variables.
    /// These [Wire]s have two outer dimensions:
    /// - 0: current/latched variables
    /// - 1: next variables
    private: Wire<D>,

    /// The atoms of this module.
    /// The atoms must be stored in a *consistent* linear order
    /// as defined in the reactive modules paper.
    /// TODO: implement `new`/`new_unchecked` methods
    atoms: Vec<Atom<D, I>>,
}
