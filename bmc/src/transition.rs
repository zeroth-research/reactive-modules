use base::Interface;
use base::Term;

/// Relation between two (ordered) sets of variables
struct Transition<D, I> {
    intf_in: Interface<D>,
    intf_out: Interface<D>,

    transition: Vec<Term<D, I>>,
}
