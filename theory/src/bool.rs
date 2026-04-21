/*!
# Booleans and operations on booleans.

Defines the propositional theory [`Prop`]: boolean matrices and the
standard logical operations over them.

A [`PropDType`] value describes the *type* of a boolean term as a matrix
shape `Bool(rows, cols)` — scalars are just `Bool(1, 1)`. The operations
in [`Prop`] are:

- [`Prop::Const`] — an inline matrix literal producing a boolean matrix of
  the shape declared by its write type.
- [`Prop::Not`], [`Prop::Id`] — unary, shape-preserving.
- [`Prop::And`], [`Prop::Or`], [`Prop::Xor`] — elementwise binary, both
  inputs and the output share the same shape.

`Prop` implements [`Theory`], so [`Theory::check`] type-checks a term by
validating the read/write argument shapes against the operation.

## Examples

```
use theory::Theory;
use theory::bool::{Prop, PropDType};

let t = PropDType::Bool(1, 1);

// `And` reads two scalars and writes one.
assert!(Prop::And.check::<PropDType>(&[t, t], &[t]).is_ok());

// `Not` is unary — two inputs is a type error.
assert!(Prop::Not.check::<PropDType>(&[t, t], &[t]).is_err());
```
*/

use crate::*;

#[derive(Clone, Copy, PartialEq)]
pub enum PropDType {
    Bool(usize, usize),
}

#[derive(Clone, PartialEq, Debug)]
pub enum Prop {
    Const(Vec<Vec<bool>>),
    And,
    Or,
    Xor,
    Not,
    Id,
}

impl Theory for Prop {
    type DType = PropDType;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> Result<(), String> {
        match self {
            Prop::Const(cm) => {
                if read.len() > 0 {
                    return Err("Const: cannot read values".into());
                }
                if write.len() != 1 {
                    return Err("Const: must return a single value".into());
                }
                match &write[0] {
                    PropDType::Bool(i, j) => {
                        if cm.len() != *i {
                            return Err(format!(
                                "Const: Initializer has a wrong number of rows (has {}, expected {})",
                                cm.len(),
                                *i
                            ));
                        }
                        if cm.iter().any(|row| row.len() != *j) {
                            return Err(format!(
                                "Const: some column of initializer has wrong dimension, expected {}",
                                *j
                            ));
                        }
                        Ok(())
                    }
                }
            }
            Prop::Not | Prop::Id => {
                if read.len() != 1 {
                    return Err(format!(
                        "{:?}: must read a single value, got {}",
                        self,
                        read.len()
                    ));
                }
                if write.len() != 1 {
                    return Err(format!(
                        "{:?}: must write a single value, got {}",
                        self,
                        write.len()
                    ));
                }
                if write[0] != read[0] {
                    return Err(format!(
                        "{:?}: input and output type must be the same",
                        self
                    ));
                }
                Ok(())
            }
            Prop::And | Prop::Or | Prop::Xor => {
                if read.len() != 2 {
                    return Err(format!(
                        "{:?}: must read two values, got {}",
                        self,
                        read.len()
                    ));
                }
                if write.len() != 1 {
                    return Err(format!(
                        "{:?}: must write a single value, got {}",
                        self,
                        write.len()
                    ));
                }
                if read[0] != read[1] {
                    return Err(format!("{:?}: input values must have the same type", self));
                }
                if write[0] != read[1] {
                    return Err(format!(
                        "{:?}: input and output values must have the same type",
                        self
                    ));
                }
                Ok(())
            }
        }
    }
}
