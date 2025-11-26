use std::fmt;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Type {
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

impl std::str::FromStr for Type {
    type Err = String;

    fn from_str(ty: &str) -> Result<Self, Self::Err> {
        match ty {
            "Real" => Ok(Type::Real),
            "Int" => Ok(Type::Int),
            "Bool" => Ok(Type::Bool),
            _ => {
                if let Some(dim) = ty.strip_prefix("MatInt(")
                    && let Some(inner) = dim.strip_suffix(")")
                    && let Some((m, n)) = parse_dim(inner)
                {
                    return Ok(Type::MatInt(m, n));
                }

                if let Some(dim) = ty.strip_prefix("MatReal(")
                    && let Some(inner) = dim.strip_suffix(")")
                    && let Some((m, n)) = parse_dim(inner)
                {
                    return Ok(Type::MatReal(m, n));
                }

                Err(format!("Cannot convert `{}` to Type", ty))
            }
        }
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Real => write!(f, "Real"),
            Type::Int => write!(f, "Int"),
            Type::Bool => write!(f, "Bool"),
            Type::MatReal(n, m) => write!(f, "MatReal({n}, {m})"),
            Type::MatInt(n, m) => write!(f, "MatInt({n}, {m})"),
        }
    }
}
