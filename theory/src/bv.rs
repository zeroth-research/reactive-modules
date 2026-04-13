/*!
# BitVectors of fixed length

Provides the [`BV<N>`] type (parameterized by bit-width `N`) with
operations [`Add`], [`Mul`], and [`Id`], bundled into a [`Theory`].

## Examples

```
use theory::bv::*;

let _: Types = BV::<8>().into();
let _: Operations = Add().into();
```
*/

use crate::{Type, mk_theory};

/// Bitvector type parametrized by its size
#[derive(Clone, Copy, PartialEq)]
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
