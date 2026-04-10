/*!
# Integers (mathematical)

This code makes available: TODO

## Examples
TODO

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
