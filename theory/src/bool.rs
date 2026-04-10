/*!
# Booleans and operations on booleans.

Provides the [`Bool`] type with operations [`And`], [`Or`], [`Xor`],
[`Not`], and [`Id`], bundled into a [`Theory`].

## Examples

```
use theory::bool::*;

let _: Types = Bool().into();
let _: Operations = And().into();
let _: Operations = Not().into();
```
*/

use crate::*;

// Create the theory of booleans
mk_theory!(
    Types(Bool),
    {
        And(Bool, Bool) => Bool,
        Or(Bool, Bool) => Bool,
        Xor(Bool, Bool) => Bool,
        Not(Bool) => Bool,
        Id(Bool) => Bool
    }
);
