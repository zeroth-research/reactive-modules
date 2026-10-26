use crate::sort::{MultiSort, TangentMultiSort};
use std::fmt;
//use std::ops::Mul;

/// The name of a variable (a type variable or a symbolic dimension).
/// Names are printed as `a`, `b`, ..., `z`, `a1`, `b1`, ..., `z1`, `a2`, ...
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Name(pub u8);

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

// impl<MS: MultiSort> SignatureElem<MS> {
//     /// Is this element concrete (i.e., it does not contain any variables)
//     pub fn is_concrete(&self) -> bool {
//         match self {
//             Self::S(_) | Self::Tan(_) => true,
//             Self::Mat(_, Dim::Const(_), Dim::Const(_)) => true,
//             _ => false,
//         }
//     }
// }

#[allow(non_snake_case)]
pub fn TypeVar<MS: MultiSort>(name: impl Into<Name>) -> SignatureElem<MS> {
    SignatureElem::<MS>::TypeVar(name.into())
}

#[allow(non_snake_case)]
pub fn Tan<MS: TangentMultiSort>(srt: MS) -> SignatureElem<MS> {
    SignatureElem::<MS>::Tan(srt)
}

#[allow(non_snake_case)]
pub fn Mat<MS: MultiSort>(srt: MS, m: impl Into<Dim>, n: impl Into<Dim>) -> SignatureElem<MS> {
    SignatureElem::<MS>::Mat(srt, m.into(), n.into())
}

impl<MS: MultiSort> From<MS> for SignatureElem<MS> {
    fn from(s: MS) -> Self {
        SignatureElem::S(s)
    }
}

impl From<usize> for Dim {
    fn from(n: usize) -> Self {
        Dim::Const(n)
    }
}

impl From<char> for Dim {
    fn from(c: char) -> Self {
        Dim::Sym(c.into())
    }
}

impl From<u8> for Name {
    fn from(n: u8) -> Self {
        Name(n)
    }
}

impl From<char> for Name {
    fn from(c: char) -> Self {
        assert!(
            c.is_ascii_lowercase(),
            "variable name must be a..z, got {c:?}"
        );
        Name(c as u8 - b'a')
    }
}

impl fmt::Display for Name {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let letter = char::from(b'a' + self.0 % 26);
        match self.0 / 26 {
            0 => write!(f, "{letter}"),
            round => write!(f, "{letter}{round}"),
        }
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Dim::Sym(name) => write!(f, "{name}"),
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
            SignatureElem::TypeVar(name) => write!(f, "'{name}"),
        }
    }
}

pub trait Signature<MS: MultiSort> {
    fn dom(&self) -> impl Iterator<Item = SignatureElem<MS>>;
    fn cod(&self) -> impl Iterator<Item = SignatureElem<MS>>;

    fn dom_arity(&self) -> usize;
    fn cod_arity(&self) -> usize;
}

/// Helper for substitutions in the signature (for type-checking)
enum SubstType<MS: MultiSort> {
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

/// Collect the elements of the signature (the domain followed by the codomain)
/// into a vector, shifting the names of all variables by `offset`
fn collect_shifted<S, MS>(sig: &S, offset: usize) -> Result<Vec<SignatureElem<MS>>, String>
where
    S: Signature<MS>,
    MS: MultiSort,
{
    let shift = |x: Name| {
        u8::try_from(x.0 as usize + offset)
            .map(Name)
            .map_err(|_| "Too many variables in the signatures".to_string())
    };

    sig.dom()
        .chain(sig.cod())
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
        .collect()
}

/// Find the next substitution that unifies the elements of `sig1` and `sig2`,
/// or `None` if there is no substitution left to do
fn find_subst<MS: MultiSort>(
    sig1: &[SignatureElem<MS>],
    sig2: &[SignatureElem<MS>],
) -> Option<SubstType<MS>> {
    for (e1, e2) in sig1.iter().zip(sig2) {
        match (e1, e2) {
            // for two type variables, pick the left one (it doesn't matter which one, but we
            // need to pick) -- and only if they are not the same (to ensure progress)
            (SignatureElem::TypeVar(n), SignatureElem::TypeVar(m)) => {
                if n != m {
                    return Some(SubstType::Elem(*e2, *e1));
                }
            }
            (SignatureElem::TypeVar(_), _) => return Some(SubstType::Elem(*e1, *e2)),
            (_, SignatureElem::TypeVar(_)) => return Some(SubstType::Elem(*e2, *e1)),
            (SignatureElem::Mat(_, n1, m1), SignatureElem::Mat(_, n2, m2)) => {
                for (d1, d2) in [(n1, n2), (m1, m2)] {
                    match (d1, d2) {
                        (Dim::Sym(x), Dim::Sym(y)) => {
                            if x != y {
                                return Some(SubstType::DimName(*y, *x));
                            }
                        }
                        (Dim::Sym(x), Dim::Const(y)) | (Dim::Const(y), Dim::Sym(x)) => {
                            return Some(SubstType::DimConst(*x, *y));
                        }
                        _ => (),
                    }
                }
            }
            _ => (),
        }
    }
    None
}

/// Apply the substitution `subst` to the elements `elems`
fn apply_subst<MS: MultiSort + PartialEq>(subst: &SubstType<MS>, elems: &mut [SignatureElem<MS>]) {
    // replace the dimension name `orig` with `new` in the matrix `e`
    let subst_dim = |e: &mut SignatureElem<MS>, orig: Name, new: Dim| {
        if let SignatureElem::Mat(_, m, n) = e {
            for d in [m, n] {
                if *d == Dim::Sym(orig) {
                    *d = new;
                }
            }
        }
    };

    for e in elems {
        match *subst {
            SubstType::Elem(orig, new) => {
                if *e == orig {
                    *e = new;
                }
            }
            SubstType::DimName(orig, new) => subst_dim(e, orig, Dim::Sym(new)),
            SubstType::DimConst(orig, new) => subst_dim(e, orig, Dim::Const(new)),
        }
    }
}

/// Check if signatures `sig1` and `sig2` are compatible.
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

    // XXX: This is not the most efficient implementation, we might replace it in the future.
    // The substitutions below act on both signatures at once, so they must not
    // share variables: rename the variables of `sig2` apart from those of `sig1`
    // by shifting them after the "greatest" name in `sig1`.

    // First pass: gather the greatest name of a variable in `sig1`.
    let offset = greatest_var_name(&sig1).map_or(0, |x| x.0 as usize + 1);

    // Second pass: collect the signatures, renaming the variables of `sig2`
    let mut sig1_vec = collect_shifted(&sig1, 0)?;
    let mut sig2_vec = collect_shifted(&sig2, offset)?;
    debug_assert!(sig1_vec.len() == sig2_vec.len());

    while let Some(subst) = find_subst(&sig1_vec, &sig2_vec) {
        apply_subst(&subst, &mut sig1_vec);
        apply_subst(&subst, &mut sig2_vec);
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
