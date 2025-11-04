use base::term::Term;
use std::fmt;

pub struct BuchiObligations<D, I> {
    pub invariant: Vec<Term<D, I>>,
    pub variant: Vec<Term<D, I>>,

    pub buchi: Vec<Term<D, I>>,
}

impl<D: fmt::Display, I: fmt::Display> fmt::Display for BuchiObligations<D, I> {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}
