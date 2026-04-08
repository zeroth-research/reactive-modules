use crate::mk_theory_mod;

mk_theory_mod!(
    real, Types(Real),
    Add(Real, Real) => Real,
    Mul(Real, Real) => Real,
    Id(Real) => Real
);
