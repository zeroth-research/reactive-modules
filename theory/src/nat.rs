/*!
# Natural numbers

Provides the [`Nat`] type with operations [`Add`], [`Mul`],
and [`Id`], bundled into a [`Theory`].

## Examples

```
use theory::nat::*;

let _: Types = Nat().into();
let _: Operations = Add().into();
```
*/

use crate::mk_theory;

mk_theory!(
    Types(Nat),
    {
        Add(Nat, Nat) => Nat,
        Mul(Nat, Nat) => Nat,
        Id(Nat) => Nat
    }
);
