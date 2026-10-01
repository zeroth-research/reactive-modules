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
* **Stochastic semantics.** A scheduler resolves the nondeterminism, leaving
  the embedded chain of the steps, with its measure on runs
  (`Zrth.Stochastic.Scheduler`); almost surely, a run of the chain is a
  possible run of the system. By a time change, the chain becomes a single
  process in continuous time, as for piecewise-deterministic jump processes,
  with its natural filtration, in which the steps start at stopping times
  (`Zrth.Stochastic.Continuous`). Finer views refine its jumps, which collapse
  everything happening at the same instant: the run in hybrid time — step,
  then local time (`Zrth.Stochastic.Run`) — and a round atom by atom
  (`Zrth.Stochastic.Round`). The states carry σ-algebras with measurable
  points, Borel for real valuations (`Zrth.Stochastic.Evolution`).

`Zrth.Calculus` brings trajectories of real variables back to real analysis,
for the proofs about the examples (`Zrth.Examples`).
-/
