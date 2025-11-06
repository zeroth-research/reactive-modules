use crate::wire::Wire;
use std::fmt;

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
    pub(crate) itype: I,
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

impl<D: fmt::Display, I: fmt::Display> fmt::Display for Term<D, I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const BOLD: &str = "\x1b[1m";
        const RESET: &str = "\x1b[0m";
        write!(f, "{} ", self.itype,)?;
        write!(
            f,
            "{}",
            self.write
                .iter()
                .map(|(a, _)| format!("w{}", a))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        write!(
            f,
            "; {}",
            self.read
                .iter()
                .map(|(a, _)| format!("w{}", a))
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}
