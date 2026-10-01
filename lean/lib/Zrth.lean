-- The static part of the Lean 4 code common to every translated reactive module.
import Zrth.Basic
import Zrth.Dist
import Zrth.Val
import Zrth.Calculus
import Zrth.Atom
import Zrth.Module
import Zrth.Hybrid
import Zrth.Stochastic

/-!
# Zrth

Reactive modules (after the Rust `base` crate, abstracted from wires and
terms) with a probabilistic hybrid semantics. The layers, in import order:

* **Syntax.** Variables range over manifolds, so they can flow; the state of a
  set of variables is a valuation (`Zrth.Val`). An atom controls some
  variables, with a probabilistic initialisation and update (sets of finitely
  supported distributions, `Zrth.Dist`) and a flow (a differential inclusion)
  (`Zrth.Atom`). A module is a list of atoms, ordered by their await relation,
  with its variables partitioned by visibility; modules compose in parallel
  (`Zrth.Module`).
* **Hybrid semantics.** A probabilistic hybrid system alternates random jumps
  with flows along trajectories, which cannot get out of jump regions
  (`Zrth.Hybrid.Basic`). A closed module is one: the atoms flow together,
  racing to fire in their jump regions, and a jump is a round, in which the
  atoms draw one after the other (`Zrth.Hybrid.Round`, `Zrth.Hybrid.Module`).
* **Stochastic semantics.** A scheduler resolves the nondeterminism, leaving a
  Markov chain, with its measure on runs (`Zrth.Stochastic.Scheduler`); almost
  surely, a run of the chain is a possible run of the system. Runs, rounds and
  flows are processes adapted to filtrations: the run in hybrid time
  (`Zrth.Stochastic.Run`), a round atom by atom (`Zrth.Stochastic.Round`), a
  flow in continuous time (`Zrth.Stochastic.Disc`).

`Zrth.Calculus` brings trajectories of real variables back to real analysis,
for the proofs about the examples (`Zrth.Examples`).
-/
