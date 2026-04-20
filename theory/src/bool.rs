/*!
# Booleans and operations on booleans.

TODO: add description

## Examples

```
// TODO: add examples
```
*/

use crate::*;

pub enum PropDType {
    Bool,
}

/// TODO: write "formal" semantics
pub enum Prop {
    Const(bool),
    And,
    Or,
    Xor,
    Not,
    Id,
}

impl Theory for Prop {
    type DType = PropDType;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> bool {
        match self {
            Prop::Const(_) => return read.len() == 0 && write.len() == 1,
            Prop::Not | Prop::Id => return read.len() == 1 && write.len() == 1,
            Prop::Xor => return read.len() == 2 && write.len() == 1,
            _ => return write.len() == 1,
        }
    }
}
