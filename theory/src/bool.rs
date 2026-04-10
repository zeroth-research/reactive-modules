/*!
# Booleans and operations on booleans.

This code makes available: TODO

## Examples
TODO

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
