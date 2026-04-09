use crate::{Type, mk_theory_mod};

/// Bitvector type parametrized by its size
#[derive(Clone)]
pub struct BV<const N: usize>();
impl<const N: usize> Type for BV<N> {}

mk_theory_mod!(
    [const N: usize]
    bv, Types(BV => BV<N>),
    Add(BV<N>, BV<N>) => BV<N>,
    Mul(BV<N>, BV<N>) => BV<N>,
    Id(BV<N>) => BV<N>
);
