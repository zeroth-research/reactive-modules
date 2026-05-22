use crate::bv::BV;
use crate::lia::LIA;
use crate::lra::LRA;
use crate::{Theory, bv, lia, lra};

enum Type {
    Bool([usize; 2]),
    Real([usize; 2]),
    Int([usize; 2]),
    BV(usize, [usize; 2]),
}

impl From<bv::Type> for Type {
    fn from(value: bv::Type) -> Self {
        match value {
            bv::Type::BV(bw, shape) => Type::BV(bw, shape),
        }
    }
}

impl From<lia::Type> for Type {
    fn from(value: lia::Type) -> Self {
        match value {
            lia::Type::Int(shape) => Type::Int(shape),
            lia::Type::Bool(shape) => Type::Bool(shape),
        }
    }
}

impl From<lra::Type> for Type {
    fn from(value: lra::Type) -> Self {
        match value {
            lra::Type::Bool(shape) => Type::Bool(shape),
            lra::Type::Real(shape) => Type::Real(shape),
        }
    }
}

impl TryFrom<Type> for bv::Type {
    type Error = String;

    fn try_from(value: Type) -> Result<Self, Self::Error> {
        match value {
            Type::BV(bw, shape) => Ok(bv::Type::BV(bw, shape)),
            _ => Err("invalid cast".to_string()),
        }
    }
}

impl TryFrom<Type> for lia::Type {
    type Error = String;

    fn try_from(value: Type) -> Result<Self, Self::Error> {
        match value {
            Type::Bool(shape) => Ok(lia::Type::Bool(shape)),
            Type::Int(shape) => Ok(lia::Type::Int(shape)),
            _ => Err("invalid cast".to_string()),
        }
    }
}

impl TryFrom<Type> for lra::Type {
    type Error = String;

    fn try_from(value: Type) -> Result<Self, Self::Error> {
        match value {
            Type::Bool(shape) => Ok(lra::Type::Bool(shape)),
            Type::Real(shape) => Ok(lra::Type::Real(shape)),
            _ => Err("invalid cast".to_string()),
        }
    }
}

#[allow(dead_code)]
#[allow(clippy::upper_case_acronyms)]
enum Any {
    LRA(LRA),
    LIA(LIA),
    BV(BV),
}

struct TryFrom2<A>
where
    A: TryInto<Type>,
{
    a: A,
}

impl<A: TryInto<Type>> TryFrom2<A> {
    #[allow(dead_code)]
    fn new(a: A) -> Self {
        Self { a }
    }
}

impl<A: TryInto<Type>> TryFrom<TryFrom2<A>> for lra::Type {
    type Error = String;

    fn try_from(value: TryFrom2<A>) -> Result<Self, Self::Error> {
        let b: Type = value.a.try_into().map_err(|_| "invalid cast")?;
        let c: lra::Type = b.try_into().map_err(|e: String| e.to_string())?;
        Ok(c)
    }
}

impl<A: TryInto<Type>> TryFrom<TryFrom2<A>> for lia::Type {
    type Error = String;

    fn try_from(value: TryFrom2<A>) -> Result<Self, Self::Error> {
        let b: Type = value.a.try_into().map_err(|_| "invalid cast")?;
        let c: lia::Type = b.try_into().map_err(|e: String| e.to_string())?;
        Ok(c)
    }
}

impl<A: TryInto<Type>> TryFrom<TryFrom2<A>> for bv::Type {
    type Error = String;

    fn try_from(value: TryFrom2<A>) -> Result<Self, Self::Error> {
        let b: Type = value.a.try_into().map_err(|_| "invalid cast")?;
        let c: bv::Type = b.try_into().map_err(|e: String| e.to_string())?;
        Ok(c)
    }
}

impl Theory for Any {
    type DType = Type;

    fn check<R, W, D>(&self, read: R, write: W) -> Result<(), String>
    where
        D: TryInto<Type>,
        R: IntoIterator<Item = D>,
        W: IntoIterator<Item = D>,
    {
        let read = read.into_iter().map(TryFrom2::new);
        let write = write.into_iter().map(TryFrom2::new);
        match &self {
            Any::LRA(itype) => itype.check(read, write),
            Any::LIA(itype) => itype.check(read, write),
            Any::BV(itype) => itype.check(read, write),
        }
    }
}
