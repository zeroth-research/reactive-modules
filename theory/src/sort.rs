/// A collection of sorts.
///
/// Use the name `MultiSort` rather than `Sort`
/// because 1) when dumping to Lean, we cannot use `Sort` as that is
/// a keyword in Lean 2) it better reflects what it is (it is not a single sort).
pub trait MultiSort: Copy {
    const NAME: &'static str;

    /// Return `true` if `self` represents a tangent space.
    /// For runtime checks.
    /// XXX: do we need this?
    fn is_tangent_sort() -> bool {
        false
    }
}

/// MultiSort that supports tangents
pub trait TangentMultiSort: MultiSort {
    fn is_tangent_sort() -> bool {
        true
    }

    /// Get the tangent of a sort
    #[allow(non_snake_case)]
    fn T(&self) -> Self;
}

//struct Mat<S: MultiSort, const n: usize, const m: usize> {
struct Mat<S: MultiSort> {
    elem_sort: S,
    m: usize,
    n: usize,
}
