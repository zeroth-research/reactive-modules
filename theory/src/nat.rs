/*!
# Natural numbers

TODO: add description

## Examples

```
// TODO: add examples
```
*/

use crate::*;

pub enum NatDType {
    Nat,
}

/// TODO: write "formal" semantics
pub enum Nat {
    Const(String),
    Add,
    Mul,
    Id,
}

impl Theory for Nat {
    type DType = NatDType;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> bool {
        match self {
            // TODO: check `s` are all digits
            Nat::Const(s) => return read.len() == 0 && write.len() == 1 && !s.is_empty(),
            Nat::Id => return read.len() == 1 && write.len() == 1,
            Nat::Add | Nat::Mul => return read.len() == 2 && write.len() == 1,
        }
    }
}
