"""What every backend reads of a reactive module, besides its terms.

`sorts`: the sort and shape of a wire, and the Lean type each prints as.
`layout`: where a wire's elements sit -- tuple accessors, and the flat
element order every scalar encoding shares. `values`: a constant as the
exact number every backend states. `ops`: the op table, one row per theory
variant and one column per backend. Nothing here imports cvc5, and nothing
here imports a backend.
"""
