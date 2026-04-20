/*!
# Real numbers

TODO: add description

## Examples

```
// TODO: add examples
```
*/

use crate::*;

pub enum RealDType {
    Real,
}

/// TODO: write "formal" semantics
pub enum Real {
    Const(String),
    Add,
    Mul,
    Id,
}

impl Theory for Real {
    type DType = RealDType;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> bool {
        match self {
            // TODO: check `s` are all digits
            Real::Const(s) => return read.len() == 0 && write.len() == 1 && !s.is_empty(),
            Real::Id => return read.len() == 1 && write.len() == 1,
            Real::Add | Real::Mul => return read.len() == 2 && write.len() == 1,
        }
    }
}
