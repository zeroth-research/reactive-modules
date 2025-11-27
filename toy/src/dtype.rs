use std::fmt;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DType {
    // numbers
    Real,
    Int,
    // booleans
    Bool,
    // matrices
    MatReal(usize, usize),
    MatInt(usize, usize),
}

fn parse_dim(inner: &str) -> Option<(usize, usize)> {
    let mut parts = inner.split(',');
    if let (Some(n), Some(m)) = (parts.next(), parts.next()) {
        if parts.next().is_some() {
            return None;
        }
        if let (Ok(m), Ok(n)) = (m.parse(), n.parse()) {
            return Some((m, n));
        }
    }

    None
}

impl std::str::FromStr for DType {
    type Err = String;

    fn from_str(ty: &str) -> Result<Self, Self::Err> {
        match ty {
            "Real" => Ok(DType::Real),
            "Int" => Ok(DType::Int),
            "Bool" => Ok(DType::Bool),
            _ => {
                if let Some(dim) = ty.strip_prefix("MatInt(")
                    && let Some(inner) = dim.strip_suffix(")")
                    && let Some((m, n)) = parse_dim(inner)
                {
                    return Ok(DType::MatInt(m, n));
                }

                if let Some(dim) = ty.strip_prefix("MatReal(")
                    && let Some(inner) = dim.strip_suffix(")")
                    && let Some((m, n)) = parse_dim(inner)
                {
                    return Ok(DType::MatReal(m, n));
                }

                Err(format!("Cannot convert `{}` to DType", ty))
            }
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::Real => write!(f, "Real"),
            DType::Int => write!(f, "Int"),
            DType::Bool => write!(f, "Bool"),
            DType::MatReal(n, m) => write!(f, "MatReal({n}, {m})"),
            DType::MatInt(n, m) => write!(f, "MatInt({n}, {m})"),
        }
    }
}
