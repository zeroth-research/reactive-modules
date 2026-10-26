use crate::sort::{MultiSort, TangentMultiSort};
use std::collections::HashMap;
use std::fmt;
//use std::ops::Mul;

type Name = u8;

// A constant or symbolic name of a dimension in matrix
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Dim {
    // symbolic name, e.g., `N` in `Mat(Int, N, 5)`
    Sym(Name),
    // concrete constant, e.g., `5` in `Mat(Int, N, 5)`
    Const(usize),
}

/// An element of a signature: In A x B -> C`, elements are A, B, C.
/// It is either a sort, its tangent, a matrix sort (that we handle specially
/// because we need to reason about the dimensions) or a type variable.
/// TODO: should be go with tensors instead of with a matrix?
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SignatureElem<MS: MultiSort> {
    // sort
    S(MS),
    // tangent space of a sort
    Tan(MS),
    // dimension variable (natural number)
    // NatVar(Name),
    // matrix of element type `MS` and dimensions of type variables
    Mat(MS, Dim, Dim),
    // type variable
    TypeVar(Name),
}

#[allow(non_snake_case)]
impl<MS: MultiSort> SignatureElem<MS> {
    pub fn mk_S(srt: MS) -> SignatureElem<MS> {
        Self::S(srt)
    }

    pub fn mk_TypeVar(name: Name) -> SignatureElem<MS> {
        Self::TypeVar(name)
    }

    //pub fn mk_NatVar(name: Name) -> SignatureElem<MS> {
    //    todo!()
    //}

    pub fn mk_Mat(srt: MS, m: Dim, n: Dim) -> SignatureElem<MS> {
        Self::Mat(srt, m, n)
    }

    /// Is this element concrete (i.e., it does not contain any variables)
    pub fn is_concrete(&self) -> bool {
        match self {
            Self::S(_) | Self::Tan(_) => true,
            Self::Mat(_, Dim::Const(_), Dim::Const(_)) => true,
            _ => false,
        }
    }
}

#[allow(non_snake_case)]
impl<MS: TangentMultiSort> SignatureElem<MS> {
    pub fn mk_Tan(srt: MS) -> SignatureElem<MS> {
        Self::Tan(srt)
    }
}

// Variables are printed as `a`, `b`, ..., `z`, `a1`, `b1`, ..., `z1`, `a2`, ...
fn fmt_var_name(f: &mut fmt::Formatter<'_>, name: Name) -> fmt::Result {
    let letter = char::from(b'a' + name % 26);
    match name / 26 {
        0 => write!(f, "{letter}"),
        round => write!(f, "{letter}{round}"),
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Sym(name) => fmt_var_name(f, *name),
            Dim::Const(n) => write!(f, "{n}"),
        }
    }
}

impl<MS: MultiSort + fmt::Display> fmt::Display for SignatureElem<MS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignatureElem::S(srt) => write!(f, "{srt}"),
            SignatureElem::Tan(srt) => write!(f, "Tan({srt})"),
            SignatureElem::Mat(srt, m, n) => write!(f, "Mat({srt}, {m}, {n})"),
            SignatureElem::TypeVar(name) => {
                f.write_str("'")?;
                fmt_var_name(f, *name)
            }
        }
    }
}

pub trait Signature<MS: MultiSort> {
    fn dom(&self) -> impl Iterator<Item = SignatureElem<MS>>;
    fn cod(&self) -> impl Iterator<Item = SignatureElem<MS>>;

    fn dom_arity(&self) -> usize;
    fn cod_arity(&self) -> usize;

    fn is_concrete(&self) -> bool {
        self.dom().all(|e| e.is_concrete()) && self.cod().all(|e| e.is_concrete())
    }
}

/// Helper for substitutions in the signature
enum SubstType<MS: MultiSort> {
    None,
    // elem with elem
    Elem(SignatureElem<MS>, SignatureElem<MS>),
    // variable name with name
    //VarName(Name, Name),
    // dimension name with name
    DimName(Name, Name),
    // name with constant
    DimConst(Name, usize),
}

/// The greatest name of a variable (type variable or dimension name) in the signature,
/// or `None` if the signature has no variables
fn greatest_var_name<S, MS>(sig: &S) -> Option<Name>
where
    S: Signature<MS>,
    MS: MultiSort,
{
    let mut greatest: Option<Name> = None;
    for e in sig.dom().chain(sig.cod()) {
        match e {
            SignatureElem::TypeVar(x) => greatest = greatest.max(Some(x)),
            SignatureElem::Mat(_, m, n) => {
                for d in [m, n] {
                    if let Dim::Sym(x) = d {
                        greatest = greatest.max(Some(x));
                    }
                }
            }
            _ => (),
        }
    }
    greatest
}

/// Check if signature `sig2` is compatible with `sig1`.
/// The variables of `sig1` and `sig2` are different, even if they have the same names.
pub fn typecheck<S, MS>(sig1: S, sig2: S) -> Result<(), String>
where
    S: Signature<MS>,
    MS: MultiSort + PartialEq + fmt::Display,
{
    if sig1.dom_arity() != sig2.dom_arity() {
        return Err(format!(
            "Signatures domains have different arity ({} != {})",
            sig1.dom_arity(),
            sig2.dom_arity()
        ));
    }

    if sig1.cod_arity() != sig2.cod_arity() {
        return Err(format!(
            "Signatures codomains have different arity ({} != {})",
            sig1.cod_arity(),
            sig2.cod_arity()
        ));
    }

    // XXX: This is not the most efficient implementation, it is temporary
    // The substitutions below act on both signatures at once, so they must not
    // share variables: rename the variables of `sig2` apart from those of `sig1`
    // by shifting them after the greatest name in `sig1`.
    // First pass: gather the greatest name of a variable in `sig1`.
    let offset = greatest_var_name(&sig1).map_or(0, |x| x as usize + 1);
    let shift = |x: Name| {
        Name::try_from(x as usize + offset)
            .map_err(|_| "Too many variables in the signatures".to_string())
    };

    // Second pass: collect the signatures, renaming the variables of `sig2`
    let mut sig1_vec: Vec<SignatureElem<MS>> = sig1.dom().chain(sig1.cod()).collect();
    let mut sig2_vec: Vec<SignatureElem<MS>> = sig2
        .dom()
        .chain(sig2.cod())
        .map(|mut e| {
            match &mut e {
                SignatureElem::TypeVar(x) => *x = shift(*x)?,
                SignatureElem::Mat(_, m, n) => {
                    for d in [m, n] {
                        if let Dim::Sym(x) = d {
                            *x = shift(*x)?;
                        }
                    }
                }
                _ => (),
            }
            Ok(e)
        })
        .collect::<Result<_, String>>()?;
    debug_assert!(sig1_vec.len() == sig2_vec.len());

    loop {
        let mut subst: SubstType<MS> = SubstType::None;
        // find the next substitution
        'gather: for (e1, e2) in sig1_vec.iter().zip(sig2_vec.iter()) {
            match (e1, e2) {
                // for two type variables, pick the left one (it doesn't matter which one, but we
                // need to pick) -- and only if they are not the same (to ensure progress)
                (SignatureElem::TypeVar(n), SignatureElem::TypeVar(m)) => {
                    if *n != *m {
                        subst = SubstType::Elem(*e2, *e1);
                        break;
                    }
                }
                (SignatureElem::TypeVar(n), _) => {
                    subst = SubstType::Elem(*e1, *e2);
                    break;
                }
                (_, SignatureElem::TypeVar(n)) => {
                    subst = SubstType::Elem(*e2, *e1);
                    break;
                }
                (SignatureElem::Mat(_, n1, m1), SignatureElem::Mat(_, n2, m2)) => {
                    for (d1, d2) in [(n1, n2), (m1, m2)] {
                        match (d1, d2) {
                            (Dim::Sym(x), Dim::Sym(y)) => {
                                if *x != *y {
                                    subst = SubstType::DimName(*y, *x);
                                    break 'gather;
                                }
                            }
                            (Dim::Sym(x), Dim::Const(y)) => {
                                subst = SubstType::DimConst(*x, *y);
                                break 'gather;
                            }
                            (Dim::Const(y), Dim::Sym(x)) => {
                                subst = SubstType::DimConst(*x, *y);
                                break 'gather;
                            }
                            _ => (),
                        }
                    }
                }
                _ => (),
            }
        }

        // do the substitutions
        match subst {
            SubstType::None => break,
            SubstType::Elem(orig, new) => {
                for e in &mut sig1_vec.iter_mut().chain(sig2_vec.iter_mut()) {
                    if *e == orig {
                        *e = new;
                    }
                }
            }
            SubstType::DimName(orig, new) => {
                for e in &mut sig1_vec.iter_mut().chain(sig2_vec.iter_mut()) {
                    if let SignatureElem::Mat(s, m, n) = *e {
                        match (m, n) {
                            (Dim::Sym(x), Dim::Sym(y)) if x == orig && y == orig => {
                                *e = SignatureElem::Mat(s, Dim::Sym(new), Dim::Sym(new))
                            }
                            (Dim::Sym(x), rhs) if x == orig => {
                                *e = SignatureElem::Mat(s, Dim::Sym(new), rhs)
                            }
                            (lhs, Dim::Sym(y)) if y == orig => {
                                *e = SignatureElem::Mat(s, lhs, Dim::Sym(new))
                            }
                            _ => (),
                        }
                    }
                }
            }
            SubstType::DimConst(orig, new) => {
                for e in &mut sig1_vec.iter_mut().chain(sig2_vec.iter_mut()) {
                    if let SignatureElem::Mat(s, m, n) = *e {
                        match (m, n) {
                            (Dim::Sym(x), Dim::Sym(y)) if x == orig && y == orig => {
                                *e = SignatureElem::Mat(s, Dim::Const(new), Dim::Const(new))
                            }
                            (Dim::Sym(x), rhs) if x == orig => {
                                *e = SignatureElem::Mat(s, Dim::Const(new), rhs)
                            }
                            (lhs, Dim::Sym(y)) if y == orig => {
                                *e = SignatureElem::Mat(s, lhs, Dim::Const(new))
                            }
                            _ => (),
                        }
                    }
                }
            }
        }
    }

    // check if the two signatures are the same
    for (e1, e2) in sig1_vec.iter().zip(sig2_vec.iter()) {
        if e1 != e2 {
            return Err(format!(
                "Failed unifying `{}` with `{}` on `{e1}` and `{e2}`",
                signature_to_string(&sig1),
                signature_to_string(&sig2)
            ));
        }
    }

    Ok(())
}

// Formats the signature as, e.g., `(A, B) -> C`: a domain or codomain that is
// not a single element is printed as a tuple
fn signature_to_string<MS, S>(sig: &S) -> String
where
    S: Signature<MS>,
    MS: MultiSort + fmt::Display,
{
    let elems_to_string = |elems: Vec<String>| match elems.as_slice() {
        [elem] => elem.clone(),
        _ => format!("({})", elems.join(", ")),
    };
    let dom = sig.dom().map(|e| e.to_string()).collect();
    let cod = sig.cod().map(|e| e.to_string()).collect();
    format!("{} -> {}", elems_to_string(dom), elems_to_string(cod))
}
