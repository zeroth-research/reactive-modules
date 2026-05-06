/*!
# Booleans and operations on booleans.

Defines the propositional theory [`Prop`]: boolean matrices and the
standard logical operations over them.

A [`DType`] value describes the *type* of a boolean term as a matrix
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
use theory::bool::{Prop, DType};

let t = Type::Bool(1, 1);

// `And` reads two scalars and writes one.
assert!(Prop::And.check::<DType>(&[t, t], &[t]).is_ok());

// `Not` is unary — two inputs is a type error.
assert!(Prop::Not.check::<DType>(&[t, t], &[t]).is_err());
```
*/

use crate::*;
use std::fmt;

#[derive(Clone, Copy, PartialEq, Debug, Eq)]
pub struct Bool(pub usize, pub usize);

// TODO: impl DType
impl Bool {
    pub fn shape(&self) -> (usize, usize) {
        (self.0, self.1)
    }
}

impl fmt::Display for Bool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bool({}, {})", self.0, self.1)
    }
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
    type DType = Bool;

    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a Bool>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let mut read = read.into_iter();
        let mut write = write.into_iter();
        match self {
            Prop::Const(cm) => {
                if read.next().is_some() {
                    return Err("Const: cannot read values".into());
                }
                let Bool(i, j) = write_nxt(&mut write, 0)?;
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
                if write.next().is_some() {
                    return Err("Const: returns > 1 value".into());
                }
                Ok(())
            }
            Prop::Not => {
                let (r, w) = (read_nxt(&mut read, 0)?, write_nxt(&mut write, 0)?);
                if *r != *w {
                    return Err(format!(
                        "{:?}: input and output type must be the same",
                        self
                    ));
                }
                if read.next().is_some() {
                    return Err(format!("{:?}: must read a single value (reads more)", self));
                }
                if write.next().is_some() {
                    return Err(format!(
                        "{:?}: must write a single value (writes more)",
                        self
                    ));
                }
                Ok(())
            }
            Prop::And | Prop::Or | Prop::Xor => {
                let w1 = write_nxt(&mut write, 0)?;
                let (r1, r2, None) = (
                    read_nxt(&mut read, 0)?,
                    read_nxt(&mut read, 1)?,
                    read.next(),
                ) else {
                    return Err(format!("{:?}: must read exactly two values", self));
                };
                if r1 != r2 {
                    return Err(format!("{:?}: input values must have the same type", self));
                }
                if w1 != r1 {
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
