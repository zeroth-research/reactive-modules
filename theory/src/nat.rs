/*!
# Natural numbers

This code makes available: TODO

## Examples
TODO

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
