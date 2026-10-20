"""`--infer fbk-proveit`: certifying through `lean-ltl-certifying`.

The route hands the module to that repository's `proveit.py`, which turns a
Lean model into a VMT transition system (`lean2vmt`), model-checks it with
ic3ia, and renders the invariant it finds as a Lean certificate about the
model. This package is everything verith does around that:

* `na` -- the model, a `Cslib.Automata.NA` in the syntax `lean2vmt` reads,
  and the checks that a module has one;
* `to_bool` -- a cvc5 term as the `Bool`-valued Lean that model is written in;
* `bridge` -- the theorem that the model is the module, without which the
  certificate would be about the model alone;
* `proveit` -- the checkout and toolchain checks, the run of `proveit.py`,
  and the certificate it returns, made into the project's own.

The route's row is `infer_route/fbk_proveit.py`.
"""
