use std::fmt;

pub mod bool;
pub mod bv;
pub mod float;
pub mod int;
pub mod lia;
pub mod python;
pub mod real;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CmpOp {
    Le,
    Lt,
    Ge,
    Gt,
    Eq,
    Ne,
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

pub trait MatrixType: PartialEq {
    fn shape(&self) -> (usize, usize);
}

impl CmpOp {
    pub fn type_check<'a, T, D, R, W>(&self, read: R, write: W) -> Result<(), String>
    where
        T: MatrixType + fmt::Debug + 'a,
        &'a T: TryInto<&'a bool::Bool, Error = ()>,
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

impl MatrixType for bool::Bool {
    fn shape(&self) -> (usize, usize) {
        (self.0, self.1)
    }
}
impl MatrixType for int::Int {
    fn shape(&self) -> (usize, usize) {
        (self.0, self.1)
    }
}
impl MatrixType for float::Float {
    fn shape(&self) -> (usize, usize) {
        (self.0, self.1)
    }
}
impl MatrixType for real::Real {
    fn shape(&self) -> (usize, usize) {
        (self.0, self.1)
    }
}
impl MatrixType for bv::BV {
    fn shape(&self) -> (usize, usize) {
        self.shape()
    }
}
impl MatrixType for lia::Type {
    fn shape(&self) -> (usize, usize) {
        self.shape()
    }
}
impl MatrixType for python::Type {
    fn shape(&self) -> (usize, usize) {
        self.shape()
    }
}

pub trait Theory {
    // TODO: in torch, from where we took this name (I think), dtype refers to
    // the type of the element in the tensor (*d*ata type). Maybe we should
    // consider renaming this to "Types" or something, to avoid confusion.
    type DType;

    fn type_check<'a, R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<&'a Self::DType>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
        Self::DType: 'a;
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
