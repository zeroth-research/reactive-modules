/*!
# Linear integer arithmetic

TODO: add description

## Examples

```
// TODO: add examples
```
*/

use crate::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum LIADType {
    Int(usize, usize),
    Bool(usize, usize),
}

impl From<int::IntDType> for LIADType {
    fn from(t: int::IntDType) -> Self {
        let int::IntDType::Int(i, j) = t;
        LIADType::Int(i, j)
    }
}

impl TryFrom<LIADType> for int::IntDType {
    type Error = ();
    fn try_from(lia_t: LIADType) -> Result<int::IntDType, Self::Error> {
        match lia_t {
            LIADType::Int(i, j) => Ok(int::IntDType::Int(i, j)),
            _ => Err(()),
        }
    }
}

impl TryFrom<LIADType> for bool::PropDType {
    type Error = ();
    fn try_from(lia_t: LIADType) -> Result<bool::PropDType, Self::Error> {
        match lia_t {
            LIADType::Bool(i, j) => Ok(bool::PropDType::Bool(i, j)),
            _ => Err(()),
        }
    }
}

impl From<bool::PropDType> for LIADType {
    fn from(t: bool::PropDType) -> Self {
        let bool::PropDType::Bool(i, j) = t;
        LIADType::Bool(i, j)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CmpOp {
    Le,
    Lt,
    Ge,
    Gt,
    Eq,
    Ne,
}

#[derive(Clone, PartialEq, Debug)]
pub enum LIA {
    Const(Vec<Vec<i64>>),
    Bool(bool::Prop),
    Cmp(CmpOp),
    Ite,
    // A*x + B where `A` and `B` are constants
    Linear(Vec<Vec<i64>>, Vec<Vec<i64>>),
    ReLU,
    Id,
}

impl Theory for LIA {
    type DType = LIADType;

    fn _check(&self, read: &[Self::DType], write: &[Self::DType]) -> Result<(), String> {
        match self {
            LIA::Const(cm) => {
                if read.len() > 0 {
                    return Err("Const: cannot read values".into());
                }
                if write.len() != 1 {
                    return Err("Const: must return a single value".into());
                }
                match write[0] {
                    LIADType::Int(i, j) => {
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
                    LIADType::Bool(_, _) => {
                        return Err("Const must be integer matrix, not boolean".into());
                    }
                }
            }
            LIA::Bool(op) => {
                //if *op == bool::Prop::MatMul {
                //    return Err("MatMul in LIA is forbiden, use `ScalMul` instead".into());
                //}

                // Do not allow different Id operations, we want to have ony one
                // XXX: move Id to the level of Terms or Theory trait? So that IType would
                // be "term generators"? It would make things cleaner and closer to the
                // theory of categorical circuits.
                if *op == bool::Prop::Id {
                    return Err("`Prop::Id` is forbiden in LIA, use `LIA::Id` instead".into());
                }
                op.check(read, write)
            }
            LIA::Cmp(op) => match op {
                CmpOp::Le | CmpOp::Lt | CmpOp::Eq | CmpOp::Ne | CmpOp::Ge | CmpOp::Gt => {
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

                    match write[0] {
                        LIADType::Bool(1, 1) => Ok(()),
                        _ => Err(format!(
                            "{:?}: input and output values must have the same type",
                            self
                        )),
                    }
                }
            },
            LIA::Id => {
                if read.len() != 1 {
                    return Err(format!(
                        "{:?}: must read one value, got {}",
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
                if read[0] != write[0] {
                    return Err(format!(
                        "{:?}: input and output must have the same type",
                        self
                    ));
                }
                Ok(())
            }
            LIA::Ite => {
                if read.len() != 3 {
                    return Err(format!(
                        "{:?}: must read three values, got {}",
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
                if read[1] != read[2] {
                    return Err(format!(
                        "{:?}: 2nd and 3rd inputs must have the same type",
                        self
                    ));
                }

                match write[0] {
                    LIADType::Bool(1, 1) => Ok(()),
                    _ => Err(format!(
                        "{:?}: input and output values must have the same type",
                        self
                    )),
                }
            }
            LIA::Linear(a, b) => {
                if read.len() != 1 {
                    return Err(format!(
                        "{:?}: must read one value, got {}",
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

                // check A and B constants
                let a_rows = a.len();
                if a_rows == 0 {
                    return Err(format!("{:?}: `A` is empty", self));
                }
                let a_cols = a[0].len();
                if a.iter().any(|row| row.len() != a_cols) {
                    return Err(format!(
                        "{:?}: `A` has invalid dimensions, rows have different lengths",
                        self
                    ));
                }

                let b_rows = b.len();
                let mut b_cols: usize = 0;
                if b_rows != 0 {
                    b_cols = b[0].len();
                    if b.iter().any(|row| row.len() != b_cols) {
                        return Err(format!(
                            "{:?}: `A` has invalid dimensions, rows have different lengths",
                            self
                        ));
                    }

                    if b_rows != 1 && b_cols != 1 {
                        return Err(format!(
                            "{:?}: `B` has to be a vector, got matrix {}x{}",
                            self, b_rows, b_cols
                        ));
                    }
                }

                match (&read[0], &write[0]) {
                    (LIADType::Int(d1, d2), LIADType::Int(d3, d4)) => {
                        if *d2 != a_rows {
                            return Err(format!(
                                "{:?}: mismatch in inner dimensions of `A` and `x`: A has {}x{}, x has {}x{}",
                                self, d1, d2, a_rows, a_rows
                            ));
                        }
                        // `A*x` is a a_rows x d2 matrix, `B` has to have these dimensions (if non-empty)
                        if b_rows > 0 && (a_rows != b_rows || *d2 != b_cols) {
                            return Err(format!(
                                "{:?}: A*x has dimension {}x{} while B has {}x{}",
                                self, a_rows, d2, b_rows, b_cols
                            ));
                        }
                        if a_rows != *d3 || *d2 != *d4 {
                            return Err(format!(
                                "{:?}: bad output matrix dimensions, expected {}x{} but got {}x{}",
                                self, a_rows, d2, d3, d4
                            ));
                        }
                    }
                    // TODO: should we allow also boolean matrices?
                    _ => return Err(format!("{:?}: input and output must be int matrices", self)),
                }
                Ok(())
            }
            LIA::ReLU => {
                if read.len() != 1 {
                    return Err(format!(
                        "{:?}: must read one value, got {}",
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
                if read[0] != write[0] {
                    return Err(format!(
                        "{:?}: input and output must have the same type",
                        self
                    ));
                }

                match write[0] {
                    LIADType::Int(_, _) => Ok(()),
                    _ => Err(format!(
                        "{:?}: input and output values must be int matrices",
                        self
                    )),
                }
            }
        }
    }
}
