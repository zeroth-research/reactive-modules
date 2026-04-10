/*!
# Integers (mathematical)

Provides the [`Int`] type with operations [`Add`], [`Mul`],
and [`Id`], bundled into a [`Theory`].

## Examples

```
use theory::int::*;

let _: Types = Int().into();
let _: Operations = Add().into();
```
*/

use crate::mk_theory;

mk_theory!(
    Types(Int),
    {
        Add(Int, Int) => Int,
        Mul(Int, Int) => Int,
        Id(Int) => Int
    }
);
