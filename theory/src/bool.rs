/*!
# Booleans and operations on booleans.

TODO: add description

## Examples

```
// TODO: add examples
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
