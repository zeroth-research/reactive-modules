/*!
# Bit-vectors of fixed length

TODO: add description

## Examples

```
// TODO: add examples
```
*/

use crate::*;

pub enum BVDType<const N: usize> {
    BV,
}

/// TODO: write "formal" semantics
pub enum BV<const N: usize> {
    // TODO: use bitarray
    Const(String),
    And,
    Or,
    Xor,
    Not,
    Id,
}

impl<const N: usize> Theory for BV<N> {
    type DType = BVDType<N>;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> bool {
        match self {
            BV::Const(_) => return read.len() == 0 && write.len() == 1,
            BV::Not | BV::Id => return read.len() == 1 && write.len() == 1,
            BV::Xor => return read.len() == 2 && write.len() == 1,
            _ => return write.len() == 1,
        }
    }
}
