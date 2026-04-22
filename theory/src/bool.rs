/*!
# Booleans and operations on booleans.

Defines the propositional theory [`Prop`]: boolean matrices and the
standard logical operations over them.

A [`PropDType`] value describes the *type* of a boolean term as a matrix
shape `Bool(rows, cols)` — scalars are just `Bool(1, 1)`. The operations
in [`Prop`] are:

- [`Prop::Const`] — an inline matrix literal producing a boolean matrix of
  the shape declared by its write type.
- [`Prop::Not`] — unary, shape-preserving.
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
}

impl Theory for Prop {
    type DType = PropDType;

    fn type_check<'a, R, W>(&self, read: R, write: W) -> Result<(), String>
    where
        R: IntoIterator<Item = &'a PropDType>,
        W: IntoIterator<Item = &'a PropDType>,
    {
        let mut read = read.into_iter();
        let mut write = write.into_iter();
        match self {
            Prop::Const(cm) => {
                if !(read.next() == None) {
                    return Err("Const: cannot read values".into());
                }
                if let Some(dtype) = write.next() {
                    match dtype {
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
                            if write.next() == None {
                                Ok(())
                            } else {
                                Err("Const: returns > 1 value".into())
                            }
                        }
                    }
                } else {
                    return Err("Const: must return a single value, returns none".into());
                }
            }
            Prop::Not => {
                if let (Some(r), Some(w)) = (read.next(), write.next()) {
                    if *r != *w {
                        return Err(format!(
                            "{:?}: input and output type must be the same",
                            self
                        ));
                    }
                } else {
                    return Err(format!("{:?}: must read and write a single value", self));
                }

                if read.next() != None {
                    return Err(format!("{:?}: must read a single value (reads more)", self));
                }

                if write.next() != None {
                    return Err(format!(
                        "{:?}: must write a single value (writes more)",
                        self
                    ));
                }
                Ok(())
            }
            Prop::And | Prop::Or | Prop::Xor => {
                let wn = write.next();
                if wn.is_none() {
                    return Err(format!("{:?}: must write a single value, got none", self));
                }
                let w1 = wn.unwrap();
                if let (Some(r1), Some(r2), None) = (read.next(), read.next(), read.next()) {
                    if *r1 != *r2 {
                        return Err(format!("{:?}: input values must have the same type", self));
                    }
                    if *w1 != *r1 {
                        return Err(format!(
                            "{:?}: input and output values must have the same type",
                            self
                        ));
                    }
                } else {
                    return Err(format!("{:?}: must read exactly two values", self,));
                }
                Ok(())
            }
        }
    }
}
