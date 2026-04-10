/*!
# Real numbers

Provides the [`Real`] type with operations [`Add`], [`Mul`],
and [`Id`], bundled into a [`Theory`].

## Examples

```
use theory::real::*;

let _: Types = Real().into();
let _: Operations = Add().into();
```
*/

use crate::mk_theory;

mk_theory!(
    Types(Real),
    {
        Add(Real, Real) => Real,
        Mul(Real, Real) => Real,
        Id(Real) => Real
    }
);
