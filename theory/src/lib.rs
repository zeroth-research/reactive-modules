use std::fmt;

pub mod bool;
pub mod bv;
pub mod float;
pub mod int;
pub mod lia;
pub mod python;
pub mod real;

/// Theory is a set of operations over some data types (more precisely,
/// matrices over some data types)
pub trait Theory {
    // TODO: in torch, from where we took this name (I think), dtype refers to
    // the type of the element in the tensor (*d*ata type). Maybe we should
    // consider renaming this to "Types" or something, to avoid confusion.
    type DType;

    /// Type-check if `self` can form a valid operation when reading values
    /// of type `read` and writing values of type `write` (where `read` and `write`
    /// are sequences of types, the order *does* matter).
    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a Self::DType>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
        Self::DType: 'a;
}

pub trait MatrixType: PartialEq {
    fn shape(&self) -> (usize, usize);
}

// ------------------------------------------------------------
//  Common operations
// ------------------------------------------------------------
#[derive(Clone, PartialEq, Debug, Eq)]
pub enum Arith {
    Add,
    Mul,
    Sub,
    Div,
    Mod,
    Neg,
    MatMul,
    Abs,
    Transpose,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpOp {
    Le,
    Lt,
    Ge,
    Gt,
    Eq,
    Ne,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FlowOp {
    Id,
    Ite,
}

impl fmt::Display for CmpOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CmpOp::Le => write!(f, "Le"),
            CmpOp::Lt => write!(f, "Lt"),
            CmpOp::Ge => write!(f, "Ge"),
            CmpOp::Gt => write!(f, "Gt"),
            CmpOp::Eq => write!(f, "Eq"),
            CmpOp::Ne => write!(f, "Ne"),
        }
    }
}

impl fmt::Display for Arith {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Arith::Add => write!(f, "Add"),
            Arith::Mul => write!(f, "Mul"),
            Arith::Sub => write!(f, "Sub"),
            Arith::Div => write!(f, "Div"),
            Arith::Mod => write!(f, "Mod"),
            Arith::Neg => write!(f, "Neg"),
            Arith::Abs => write!(f, "Abs"),
            Arith::MatMul => write!(f, "MatMul"),
            Arith::Transpose => write!(f, "Transpose"),
        }
    }
}

impl fmt::Display for FlowOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlowOp::Id => write!(f, "Id"),
            FlowOp::Ite => write!(f, "Ite"),
        }
    }
}

impl FlowOp {
    pub fn type_check<'a, T, D, R, W>(&self, read: R, write: W) -> Result<(), String>
    where
        T: MatrixType + fmt::Debug + 'a,
        &'a T: TryInto<&'a bool::Bool>,
        D: TryInto<&'a T>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let mut read = read.into_iter();
        let mut write = write.into_iter();
        match self {
            FlowOp::Id => {
                let (r0, None) = (read_nxt::<_, _, T>(&mut read, 0)?, read.next()) else {
                    return Err("Id: must read exactly one value".into());
                };
                let (w0, None) = (write_nxt::<_, _, T>(&mut write, 0)?, write.next()) else {
                    return Err("Id: must write exactly one value".into());
                };
                if r0 != w0 {
                    return Err(format!(
                        "Id: input and output must have the same type, got {r0:?} and {w0:?}"
                    ));
                }
                Ok(())
            }
            FlowOp::Ite => {
                let (r0, r1, r2, None) = (
                    read_nxt::<_, _, T>(&mut read, 0)?,
                    read_nxt::<_, _, T>(&mut read, 1)?,
                    read_nxt::<_, _, T>(&mut read, 2)?,
                    read.next(),
                ) else {
                    return Err("Ite: must read exactly three values".into());
                };
                let (w0, None) = (write_nxt::<_, _, T>(&mut write, 0)?, write.next()) else {
                    return Err("Ite: must write exactly one value".into());
                };
                let r0_bool: &bool::Bool = r0
                    .try_into()
                    .map_err(|_| format!("Ite: condition must be Bool(1, 1), got {r0:?}"))?;
                if r0_bool.shape() != (1, 1) {
                    return Err(format!("Ite: condition must be Bool(1, 1), got {r0:?}"));
                }
                if r1 != r2 {
                    return Err(format!(
                        "Ite: both branches must have the same type, got {r1:?} and {r2:?}"
                    ));
                }
                if w0 != r1 {
                    return Err(format!(
                        "Ite: output type {w0:?} must match branch type {r1:?}"
                    ));
                }
                Ok(())
            }
        }
    }
}

impl CmpOp {
    pub fn type_check<'a, T, D, R, W>(&self, read: R, write: W) -> Result<(), String>
    where
        T: MatrixType + fmt::Debug + 'a,
        &'a T: TryInto<&'a bool::Bool>,
        D: TryInto<&'a T>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let mut read = read.into_iter();
        let mut write = write.into_iter();
        let (r0, r1, None) = (
            read_nxt::<_, _, T>(&mut read, 0)?,
            read_nxt::<_, _, T>(&mut read, 1)?,
            read.next(),
        ) else {
            return Err(format!("{self:?}: must read exactly two values"));
        };
        let (w0, None) = (write_nxt::<_, _, T>(&mut write, 0)?, write.next()) else {
            return Err(format!("{self:?}: must write exactly one value"));
        };
        if r0 != r1 {
            return Err(format!(
                "{self:?}: inputs must have the same type, got {r0:?} and {r1:?}"
            ));
        }
        if <&T as TryInto<&bool::Bool>>::try_into(r0).is_ok() {
            return Err(format!("{self:?}: inputs cannot be Bool"));
        }
        let w_bool: &bool::Bool = w0
            .try_into()
            .map_err(|_| format!("{self:?}: output must be Bool, got {w0:?}"))?;
        if w_bool.shape() != r0.shape() {
            return Err(format!(
                "{self:?}: output shape must be {:?}, got {:?}",
                r0.shape(),
                w_bool.shape(),
            ));
        }
        Ok(())
    }
}

impl Arith {
    pub fn type_check<'a, T, D, R, W>(&self, read: R, write: W) -> Result<(), String>
    where
        T: MatrixType + fmt::Debug + 'a,
        D: TryInto<&'a T>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let mut read = read.into_iter();
        let mut write = write.into_iter();
        match self {
            Arith::Neg | Arith::Abs => {
                let (r, w) = (read_nxt(&mut read, 0)?, write_nxt(&mut write, 0)?);
                if r != w {
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

            Arith::Add | Arith::Mul | Arith::Sub | Arith::Mod | Arith::Div => {
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

            Arith::MatMul => {
                let (r1, r2, None) = (
                    read_nxt(&mut read, 0)?,
                    read_nxt(&mut read, 1)?,
                    read.next(),
                ) else {
                    return Err(format!("{:?}: must read exactly two values", self));
                };
                let (w1, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                    return Err(format!("{:?}: must write exactly one value", self));
                };

                let (d1, d2) = r1.shape();
                let (d3, d4) = r2.shape();
                let (d5, d6) = w1.shape();

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
                Ok(())
            }

            Arith::Transpose => {
                let (r, None) = (read_nxt(&mut read, 0)?, read.next()) else {
                    return Err("Transpose: must read exactly one value".into());
                };
                let (w, None) = (write_nxt(&mut write, 0)?, write.next()) else {
                    return Err("Transpose: must write exactly one value".into());
                };
                let (rm, rn) = r.shape();
                let (wm, wn) = w.shape();
                if wm != rn || wn != rm {
                    return Err(format!(
                        "Transpose: output shape ({wm}, {wn}) must be ({rn}, {rm})"
                    ));
                }
                Ok(())
            }
        }
    }
}

// Helpers for type-checking procedures

fn read_nxt<'a, R, D, T>(read: &mut R, i: usize) -> Result<&'a T, String>
where
    R: Iterator<Item = D>,
    D: TryInto<&'a T>,
{
    if let Some(d) = read.next() {
        d.try_into()
            .map_err(|_| format!("Read arg {i} not compatible with Theory"))
    } else {
        Err(format!("Read arg {i} expected, but got none"))
    }
}

fn write_nxt<'a, R, D, T>(write: &mut R, i: usize) -> Result<&'a T, String>
where
    R: Iterator<Item = D>,
    D: TryInto<&'a T>,
{
    if let Some(d) = write.next() {
        d.try_into()
            .map_err(|_| format!("Write arg {i} not compatible with Theory"))
    } else {
        Err(format!("Write arg {i} expected, but got none"))
    }
}

fn fmt_matrix<T: fmt::Display>(cm: &[Vec<T>], f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(f, "[")?;
    for (i, row) in cm.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "[")?;
        for (j, v) in row.iter().enumerate() {
            if j > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v}")?;
        }
        write!(f, "]")?;
    }
    write!(f, "]")
}
