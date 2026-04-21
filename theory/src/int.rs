/*!
# Int numbers

TODO: add description

## Examples

```
// TODO: add examples
```
*/

use crate::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum IntDType {
    Int(usize, usize),
}

#[derive(Clone, PartialEq, Debug)]
pub enum Int {
    // TODO: use bigint?
    Const(Vec<Vec<i64>>),
    Add,
    Mul,
    MatMul,
    Id,
}

impl Theory for Int {
    type DType = IntDType;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> Result<(), String> {
        match self {
            Int::Const(cm) => {
                if read.len() > 0 {
                    return Err("Const: cannot read values".into());
                }
                if write.len() != 1 {
                    return Err("Const: must return a single value".into());
                }
                match write[0] {
                    IntDType::Int(i, j) => {
                        if cm.len() != i {
                            return Err(format!(
                                "Const: Initializer has a wrong number of rows (has {}, expected {})",
                                cm.len(),
                                i
                            ));
                        }
                        if cm.iter().any(|row| row.len() != j) {
                            return Err(format!(
                                "Const: some column of initializer has wrong dimension, expected {}",
                                j
                            ));
                        }
                        Ok(())
                    }
                }
            }
            Int::Id => {
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
            Int::Add | Int::Mul => {
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
            Int::MatMul => {
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
                match (&read[0], &read[1], &write[0]) {
                    (IntDType::Int(d1, d2), IntDType::Int(d3, d4), IntDType::Int(d5, d6)) => {
                        if d2 != d3 {
                            return Err(format!(
                                "{:?}: mismatch in inner dimensions of input matrices: {} != {}",
                                self, d2, d3
                            ));
                        }
                        if d1 != d5 {
                            return Err(format!(
                                "{:?}: mismatch in first input and output dimensions: {} != {}",
                                self, d1, d5
                            ));
                        }

                        if d4 != d6 {
                            return Err(format!(
                                "{:?}: mismatch in second input and output dimensions: {} != {}",
                                self, d4, d6
                            ));
                        }
                    }
                }
                Ok(())
            }
        }
    }
}
