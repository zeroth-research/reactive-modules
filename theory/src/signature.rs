use crate::sort::{MultiSort, TangentMultiSort};
use std::{fmt, ops::Mul};
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

// A sort or a symbolic name of the elements of a matrix
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MatSort<MS: MultiSort> {
    // symbolic name, e.g., `'t` in `Mat('t, N, 5)`; it is a type variable,
    // i.e., `'t` in `'t -> Mat('t, N, 5)` is the same variable
    // (that can be instantiated only by a sort, not by a matrix or a tangent, for now)
    Sym(Name),
    // concrete sort, e.g., `Int` in `Mat(Int, N, 5)`
    S(MS),
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
    // matrix of elements of a (possibly symbolic) sort and of (possibly symbolic) dimensions
    Mat(MatSort<MS>, Dim, Dim),
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
pub fn Mat<MS: MultiSort>(
    srt: impl Into<MatSort<MS>>,
    m: impl Into<Dim>,
    n: impl Into<Dim>,
) -> SignatureElem<MS> {
    SignatureElem::<MS>::Mat(srt.into(), m.into(), n.into())
}

impl<MS: MultiSort> From<MS> for SignatureElem<MS> {
    fn from(s: MS) -> Self {
        SignatureElem::S(s)
    }
}

impl<MS: MultiSort> From<MS> for MatSort<MS> {
    fn from(s: MS) -> Self {
        MatSort::S(s)
    }
}

impl<MS: MultiSort> From<char> for MatSort<MS> {
    fn from(c: char) -> Self {
        MatSort::Sym(c.into())
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

impl<MS: MultiSort + fmt::Display> fmt::Display for MatSort<MS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatSort::Sym(name) => write!(f, "'{name}"),
            MatSort::S(srt) => write!(f, "{srt}"),
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

/// A signature given by the elements of its domain and codomain
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VecSignature<MS: MultiSort> {
    pub dom: Vec<SignatureElem<MS>>,
    pub cod: Vec<SignatureElem<MS>>,
}

impl<MS: MultiSort> Signature<MS> for VecSignature<MS> {
    fn dom(&self) -> impl Iterator<Item = SignatureElem<MS>> {
        self.dom.iter().copied()
    }

    fn cod(&self) -> impl Iterator<Item = SignatureElem<MS>> {
        self.cod.iter().copied()
    }

    fn dom_arity(&self) -> usize {
        self.dom.len()
    }

    fn cod_arity(&self) -> usize {
        self.cod.len()
    }
}

/// Helper for substitutions in the signature (for type-checking)
enum SubstType<MS: MultiSort> {
    // type variable (also as the sort of matrix elements) with elem
    Var(Name, SignatureElem<MS>),
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
            SignatureElem::Mat(s, m, n) => {
                if let MatSort::Sym(x) = s {
                    greatest = greatest.max(Some(x));
                }
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
                SignatureElem::Mat(s, m, n) => {
                    if let MatSort::Sym(x) = s {
                        *x = shift(*x)?;
                    }
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
    // is the variable `x` the sort of the elements of some matrix?
    let is_mat_sort = |x: Name| {
        sig1.iter()
            .chain(sig2)
            .any(|e| matches!(e, SignatureElem::Mat(MatSort::Sym(y), _, _) if *y == x))
    };

    for (e1, e2) in sig1.iter().zip(sig2) {
        match (e1, e2) {
            // for two type variables, pick the left one (it doesn't matter which one, but we
            // need to pick) -- and only if they are not the same (to ensure progress)
            (SignatureElem::TypeVar(n), SignatureElem::TypeVar(m)) => {
                if n != m {
                    return Some(SubstType::Var(*m, *e1));
                }
            }
            (SignatureElem::TypeVar(x), e) | (e, SignatureElem::TypeVar(x)) => {
                // The sort of matrix elements cannot be a matrix or a tangent (for now), so we
                // do not substitute them for such a variable. The elements stay different and
                // the unification fails. This also excludes `'t` vs. `Mat('t, ..)`.
                let is_sort = !matches!(e, SignatureElem::Mat(..) | SignatureElem::Tan(_));
                if is_sort || !is_mat_sort(*x) {
                    return Some(SubstType::Var(*x, *e));
                }
            }
            (SignatureElem::Mat(s1, n1, m1), SignatureElem::Mat(s2, n2, m2)) => {
                match (s1, s2) {
                    (MatSort::Sym(x), MatSort::Sym(y)) => {
                        if x != y {
                            return Some(SubstType::Var(*y, SignatureElem::TypeVar(*x)));
                        }
                    }
                    (MatSort::Sym(x), MatSort::S(s)) | (MatSort::S(s), MatSort::Sym(x)) => {
                        return Some(SubstType::Var(*x, SignatureElem::S(*s)));
                    }
                    _ => (),
                }
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
    // replace the type variable `orig` with `new` in `e`, including the sort of matrix elements
    let subst_var = |e: &mut SignatureElem<MS>, orig: Name, new: SignatureElem<MS>| match e {
        SignatureElem::TypeVar(x) if *x == orig => *e = new,
        SignatureElem::Mat(s, _, _) if *s == MatSort::Sym(orig) => {
            *s = match new {
                SignatureElem::S(srt) => MatSort::S(srt),
                SignatureElem::TypeVar(x) => MatSort::Sym(x),
                _ => unreachable!("matrix elements must be sorts (ensured by `find_subst`)"),
            }
        }
        _ => (),
    };

    for e in elems {
        match *subst {
            SubstType::Var(orig, new) => subst_var(e, orig, new),
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
