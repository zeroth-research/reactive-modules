"""What every SMT route shares: walks over cvc5 terms, the SMT-LIB printer,
the Vampire driver, solver time limits and the artifact recorder.

Nothing here imports cvc5 at import time; `cli` imports `limits`, and a bare
`verith` run never loads the solver.
"""
