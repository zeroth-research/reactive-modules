use crate::wire::Wire;

/// A single term corresponds to a single instruction
/// and has an input (`read`) and output (`write`).
///
/// Terms can be over different instruction sets (e.g., pytorch/linear guarded commands).
///
/// A list of terms represents a compute graph. A term is a node in the graph,
/// and it references the input/output edges (read/write wires).
/// [Wire]s are essentially single static assignments.
#[derive(Debug)]
pub struct Term<D, I> {
    /// The instruction to be executed by this node.
    itype: I,
    /// The outputs of this term.
    pub(crate) write: Wire<D>,
    /// The inputs to this term.
    pub(crate) read: Wire<D>,
}

impl<D, I> Term<D, I> {
    pub fn new(itype: I, write: Wire<D>, read: Wire<D>) -> Self {
        Self { itype, write, read }
    }
}
