use crate::mk_theory;

mk_theory!(
    real, Types(Real),
    Add(Real, Real) => Real,
    Mul(Real, Real) => Real,
    Id(Real) => Real
);
