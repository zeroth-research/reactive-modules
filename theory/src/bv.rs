/*!
# BitVectors of fixed length

This code makes available: TODO

## Examples
TODO

*/

use crate::{Type, mk_theory};

/// Bitvector type parametrized by its size
#[derive(Clone)]
pub struct BV<const N: usize>();
impl<const N: usize> Type for BV<N> {}

mk_theory!(
    Types([const N: usize] BV => BV<N>),
    {
        [const N: usize]
        Add(BV<N>, BV<N>) => BV<N>,
        Mul(BV<N>, BV<N>) => BV<N>,
        Id(BV<N>) => BV<N>
    }
);
