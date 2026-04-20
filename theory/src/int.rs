/*!
# Int numbers

TODO: add description

## Examples

```
// TODO: add examples
```
*/

use crate::*;

pub enum IntDType {
    Int,
}

/// TODO: write "formal" semantics
pub enum Int {
    Const(String),
    Add,
    Mul,
    Id,
}

impl Theory for Int {
    type DType = IntDType;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> bool {
        match self {
            // TODO: check `s` are all digits
            Int::Const(s) => return read.len() == 0 && write.len() == 1 && !s.is_empty(),
            Int::Id => return read.len() == 1 && write.len() == 1,
            Int::Add | Int::Mul => return read.len() == 2 && write.len() == 1,
        }
    }
}
