use crate::wire::Wire;

/// A single term corresponds to a single instruction
/// and has an input (`read`) and output (`write`).
///
/// Terms can be over different instruction sets (e.g., pytorch/linear guarded commands).
///
/// A list of terms represents a compute graph. A term is a node in the graph,
/// and it references the input/output edges (read/write wires).
pub struct Term<D, I: Instruction> {
    /// The inputs to this term.
    read: Wire<D>,
    /// The outputs of this term.
    write: Wire<D>,
    /// The instruction to be executed by this node.
    ins: I,
}

impl<D, I: Instruction> Term<D, I> {
    pub fn new(ins: I, write: Wire<D>, read: Wire<D>) -> Self {
        Self { read, write, ins }
    }
}

/// An instruction set (usually an enum of instructions) will implement this trait.
pub trait Instruction {
    /// This is an example of a function on an instruction.
    /// More likely, we want to have a type check function that can check the input/output wires.
    fn arity(&self) -> Option<usize>;
}
