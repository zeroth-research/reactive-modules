"""The certificate as Lean, and the tactics its proof is attempted with.

`render` prints the certificate's files (`Certificate/Data.lean`,
`Certificate/Certificate.lean`), its predicates through `from_smt`, which
prints a cvc5 term as a Lean expression. `tactics` plans the proof of each
obligation, from the printed predicates and from what `facts` reads off
their terms and asks cvc5.
"""
