# Design: signatures, flow, and binding

The base splits into four crates with a diamond dependency: two independent
syntax crates over one interface, meeting in a semantics crate.

```
theory      Signature, Sort, Tangent; LRA, LIA, BV
              ↑                    ↑
flow        Op<G: Signature>     reactive   Var, Atom, Module
(structural identity: OpId)      (nominal identity: Var)
              ↖                    ↗
               bind    bind : Module → Instance
                       Wire, Binding<T>
                          ↑
               python / smt / torch / interpreters
```

- **`theory`** — the one interface (`Signature`) and its base implementations.
- **`flow`** — computation syntax: `Op`, the free construction over any
  signature. Positional, wire-free, lazy.
- **`reactive`** — state syntax: `Var`, `Atom`, `Module`. Names, time roles,
  await order, visibility. Depends only on `theory` — it never sees `flow`.
- **`bind`** — the only crate that knows both worlds: elaborates a module
  into an `Instance`, minting wires. Everything semantic (evaluation, SMT,
  display, monitors) is a client of `bind`.

Seven public syntax names (`Signature`, `Tangent`, the generator enums;
`Op`; `Var`, `Atom`, `Module`), two identities, one derived coordinate, one
crossing operation. "Wires exist only below `bind`" is enforced by the
crate graph, not by documentation: the syntax crates cannot name a wire.

## Signatures

**`Signature`** is a many-sorted signature: `Signature::Sort` is the sort
set, the implementing type's values are the operation symbols
(**generators**), `check(read, write)` is the arity discipline — relational,
because generators are shape-polymorphic (no fixed arity). `LRA`, `LIA`,
`BV` are signatures; so is `flow::Op<G>` (see below). Vocabulary
stratifies: a *signature* declares **generators**; an **op** is what `flow`
builds from them. `LRA`/`LIA`/`BV` are *generator enums*, not "op enums".

Accessors: a wire's `dtype` becomes **`sort()`**; an op's `itype` becomes
**`sym()`** (the generator symbol). Not `gen()`: `gen` is a reserved
keyword in Rust 2024 (the reason `rand` renamed `Rng::gen` to `random()`),
so that accessor would need `r#gen` at every call site. (`label()` is the
fallback if `sym` reads too SMT-flavoured.)

The name **`Theory`** is reserved for the upcoming subtrait
`Theory: Signature` carrying axioms (equations between ops). Rationale:
arity checking runs at construction time everywhere; axioms feed
rewriting/equivalence machinery; uninterpreted signatures need no
empty-axioms boilerplate.

**`Sort`** is both the algebraic partner of "signature" and SMT-LIB
vocabulary (`Real`, `Int`, `(_ BitVec n)`), so both audiences read it
natively.

Structural generators remain the only obligations beyond `check`:
`Combinatorial::havoc`, `Sequential::skip`, `Differential::zero` (and
`assume`, when partiality lands) exist because `reactive` synthesizes
implicit behaviour. They lift from generators to composite ops pointwise,
but each lift is a written decision, not a freebie.

## flow: the free construction

**`Op<G: Signature>`** is the free rig category with finite biproducts over
the signature `G`, in the sense of tape diagrams — one uniform term
calculus covering dataflow (⊗) and control flow (⊕). Boundaries are
positional: `Monomial<S> = Vec<S>` (a ⊗-word), `Polynomial<S> =
Vec<Monomial<S>>` (a ⊕-word of monomials). There are no wires and no
variables in `flow`; identity of data is positional.

Term nodes (children by id — no `Box`, no recursive types):

- leaves: `Gen { sym, dom, cod }` (a generator *instantiated* at concrete
  arities — polynomials, so predicates like `eqz : S → I ⊕ I` are ordinary
  generators), `Copy(S)`, `Discard(S)`, `SymTensor`, identities, the
  ⊕-structure (`Diag`, `Codiag`, `SymSum`, units);
- compounds: `Seq`, `Sum`, `LWhisker`/`RWhisker` (tensor by monomials only —
  general ⊗ and the distributors are *derived*, faithful to the calculus).

Because internals are positional, sharing and discarding are explicit
(`Copy`/`Discard`) — the gs-monoidal presentation, and the place where
"copy is not natural (havoc)" will live as an axiom.

### The arena

Ops live in a store: `Vec` of nodes addressed by **`OpId`**, with
hash-consing (a node map) so structurally equal terms share an id, and
boundaries (`dom`/`cod`) cached per node. Consequences:

- **Construction is lazy and linear** — `Seq`/`Sum`/whiskering append one
  node; nothing distributes at construction.
- **Creation order is topological order** (children precede parents by
  construction), so no sorting and no cycle checks: acyclicity is
  structural.
- **Interpretation is one SSA pass**: a dense `Vec` indexed by `OpId` is
  the register file; each node computes its semantic object from its
  children's, once — hash-consing is global value numbering. Every backend
  (evaluator, torch, SMT, printer, normalizer) is a `for` loop over ids
  with a rule per variant.
- Ids are store-relative (unlike `Var`): an `OpId` used against the wrong
  store is the design's one new footgun — mitigate with a stamped store id
  or a brand when the code lands.

### Normal form on demand

`normalize()` produces the tape normal form: a matrix of circuits indexed
by (cod-summand, dom-summand), each cell a *formal sum* of circuits
(biproducts enrich homs in commutative monoids; the empty sum is the zero
map). This is simultaneously the equality decision procedure, the DNF, and
the guarded-command export: distributing a predicate generator yields its
*components* — the generator restricted to one output summand, i.e. exactly
an `assume` (partial, no output). Guarded commands are the normal form of
tape-typed predicates. The exponential cost of the matrix lives only here,
invoked deliberately; composition never normalizes.

The conditional, for the record — `if x == 0 then x+1 else x-1` as an op
`S → S`: `copy ; (eqz ⊗ id) ; (inc ⊕ dec) ; ∇`. Strictness makes
`(I⊕I) ⊗ S` literally the polynomial `S ⊕ S`, so the tag routes the value
with no explicit distributor node.

### Op is a Signature

`impl Signature for Op<G>`: `check` compares a proposed wiring against the
concrete boundary (functional, where a generator's check is relational —
both fit the trait). Consequences:

- `Op<Op<G>>` — the free construction is a **monad on signatures**;
  `flatten` (substitution: splice a folded op-generator in place) is its
  multiplication. Folded op-generators are the sharing mechanism; nothing
  ever expands implicitly.
- **Derived operations for free**: a library of named macros (`Ite`,
  `abs`, a neural layer) is a set of op-valued generators; `flatten` is the
  macro expander. Definitional extension without touching a theory enum.
- Anything that consumes a `Signature` — an atom slot, another op —
  accepts a composite without knowing it.

## reactive: state syntax

Purely syntactic; all static checking is sort checking plus the
order/visibility discipline. Bounded on `Signature` only.

- **`Var`** — a sort-carrying name. It mints *nothing* (no wires): a global
  creation-ordered id, `Eq`/`Hash`/`Ord` by that id. The gensym of the
  stack: it exists to distinguish.
- **`Atom<I, J, F>`** (each `: Signature`) — three labelled bindings of
  signature elements to variable lists, one per temporal role:

  | role     | writes             | reads                                  |
  |----------|--------------------|----------------------------------------|
  | `init`   | next (sort `s`)    | next of awaited vars (sort `s`)        |
  | `update` | next (sort `s`)    | latched (`reads`) + awaited next (`waits`) |
  | `delay`  | derivative (`s.T()`) | latched (`reads`) + awaited derivative (`waits`, sort `s.T()`) |

  There are no aspects and no per-entry tags: the face of every position is
  a function of (role, list). The update's latched/next distinction *is*
  the read/wait split atoms have always declared, and **delay has the same
  split**: its wait list reads awaited *derivatives* (`d(w)` of variables
  another atom flows), under the same acyclicity discipline as update's
  awaits — this is what admits derived observers such as the Lie
  derivative (`d(v) = ∇V(x) · f(x)` reads the plant's derivative fibers),
  while cyclic derivative-awaits (genuinely implicit DAEs, algebraic
  loops) stay rejected. `T` is invoked in exactly two positions: delay
  writes and delay waits. A var both read and awaited appears in both
  lists.
- **`Module`** — atoms glued by a linear order consistent with their await
  relation, plus the visibility classes (extl/intf/prvt, obs/ctrl) and the
  hiding/composition algebra, semantically as today.

`X` and `d` are gone from the core syntax: `X(x)` marked "the next face"
and roles now determine faces, so there is nothing left to mark. `d`
survives as `T` on sorts; `X` survives as the register semantics inside
`bind`. The python surface keeps `X(x)`/`d(x)` as sugar that sorts entries
into the right list.

### The sugar constructors, by example

Each atom kind is "which bundle carries nontrivial dynamics"; the sugar
constructor writes the trivial one for you — the zero section of the other
bundle. Surface syntax, target vocabulary:

**Sequential — a counter.** Dynamics in Δ; the flow is synthesized:

```python
x = Var(Real([1, 1]))
counter = Module.sequential(
    [x],
    init   = [Op(LRA.Real(zeros(1, 1)), [X(x)])],       # x' = 0
    update = [Op(LRA.Linear(I, one),    [X(x)], [x])],  # x' = x + 1
)
# synthesized flow: d(x) = 0 — the T zero section (zero_field::<Flow>)
```

**Differential — exponential decay.** Dynamics in T; the update is
synthesized:

```python
x = Var(Real([1, 1]))
decay = Module.differential(
    [x],
    init  = [Op(LRA.Real(x0),     [X(x)])],             # x' = x0
    delay = [Op(LRA.Drift(-a, 0), [d(x)], [x, dt])],    # dx = -a·x·dt
)
# synthesized update: X(x) = x — the Δ zero section (zero_field::<Step>)
```

**Hybrid — the bouncing ball.** Both fields explicit: continuous fall,
discrete reflection at the floor.

```python
p, v = Var(Real([1, 1])), Var(Real([1, 1]))
ball = Module.hybrid(
    [p, v],
    init   = [Op(LRA.Real(p0), [X(p)]), Op(LRA.Real(zeros(1, 1)), [X(v)])],
    update = [                       # the jump: reflect v at the floor
        Op(LRA.Le(),  [hit], [p, zero]),           # hit  = p ≤ 0
        Op(LRA.Ite(), [X(v)], [hit, minus_cv, v]), # v'   = hit ? -c·v : v
        Op(LRA.Id(),  [X(p)], [p]),                # p'   = p
    ],
    delay  = [                       # the flow: free fall
        Op(LRA.Drift(1, 0), [d(p)], [v, dt]),      # dp = v·dt
        Op(LRA.Drift(0,-g), [d(v)], [v, dt]),      # dv = -g·dt
    ],
)
```

The examples use **`Drift(A, b)`** — the affine flow generator
`d(y) = (A·x + b)·dt`, reading a base `x` (rank 0) *and the time form
`dt`* (rank 1), writing a fiber (rank 1). Note what it is **not**: a
degree violation. A rate is not a form — `dv = -g` would equate a 1-form
with a 0-form — and rank checking is *form-degree conservation*: every
generator conserves degree (products add it), which is exactly why the
rank-preserving `Linear` rightly refuses to cross. `Drift` conserves
degree too (0 + 1 = 1); the only primitive source of degree is the **time
form `dt`** itself — the tautological differential of the clock, either a
distinguished 1-form the module provides (minted by `bind`) or the
derivative face of an explicit clock variable (`d(t) = dt`). The same
graded product (0-form × 1-form → 1-form) is precisely the op
forward-mode autodiff emits (`∇V(x) · dx`), so one addition serves both.
LRA has none of this yet — today only `zero` writes 1-forms, so no
nontrivial flow is expressible; listed under Open items.

## bind: elaboration

The definition/elaboration split (as in HDLs): everything above is a
*definition* — checkable, composable, freely shared. Wires are what the one
crossing operation produces:

```
bind : Module → Instance
```

- **`Wire`** — the derived coordinate, minted only here:
  `(occurrence, OpId, side, index)`. The occurrence path distinguishes the
  same hash-consed op used in two atoms (definition vs instance — the
  folded-netlist distinction), so hash-consing (share syntax maximally) and
  instantiation (separate state per use) never conflict.
- **Coupling is realized at binding**: every occurrence of a variable's
  face across all atoms binds to the same wire — shared variables couple
  modules automatically; composition is purely syntactic bookkeeping.
- **Freshness is by construction**: internals are per-occurrence, and
  syntax cannot even mention another atom's internals — the shared-local-
  wire and wire-disjointness bug class (see `cannot_share_local_wires`,
  `local_coupling_is_order_independent` in `base/tests/module.rs`) becomes
  unrepresentable rather than checked.
- Wire allocation is demand-driven per var: latched if read, next if
  written or awaited, derivative if a delay writes it.
- **`Binding<T>`** — partial assignment of wires into some world; every
  consumer is an instance:

  | binding            | codomain `T`                    | totality        |
  |--------------------|---------------------------------|-----------------|
  | evaluation / SSA   | value, tensor, formula          | total           |
  | naming / display   | names                           | sparse          |
  | SMT witnesses      | solver constants                | per query       |
  | tapping / monitors | `Var` (interface-extending)     | deliberate      |

  A wire's identity is structural and unique; its *life* is one per
  interpretation. Tapping — binding a `Var` at an internal wire — is the
  inverse of hiding: it exposes an internal cut as an observable variable,
  which is how observers/monitors over internal signals will work.

## Identity

Two primitives, doing opposite jobs:

- **`Var`** (reactive) — gensym: *separates* what is equal-looking. Global,
  creation-ordered; equality is identity; creation order is the canonical
  interface order.
- **`OpId`** (flow) — value number: *unifies* what is equal-being.
  Store-scoped, dense; equality is structural equality; creation order is
  topological order.

Everything else derives: `Wire` is a coordinate over both; sorts and
generators are values; atoms and modules are composite values; ops are
positions. Nothing else in the stack mints an identifier.

## The differential structure

### Sort-level tangents

```rust
/// On the sort type: sorts closed under the tangent former.
pub trait Tangent: Sized {
    #[allow(non_snake_case)]
    fn T(self) -> Self;
}
```

The former lives on the *sort* because `reactive` needs it with no
signature at hand (a module carries three signatures sharing one sort
type). The old wire `degree` (`0`-form/`1`-form) is fully absorbed: faces
are determined by role, and the delay-writes face has sort `s.T()` — so
"derivative of an integer wire" is unwritable rather than merely useless.

A reference implementation over reals, booleans, and the trivial tangent:

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Sort {
    /// A real tensor; `rank` is the differential grade: 0 = value,
    /// 1 = first derivative, ...
    Real { shape: [usize; 2], rank: u8 },
    /// A boolean tensor: a constant sort — it cannot move during delay.
    Bool([usize; 2]),
    /// The trivial tangent: a singleton, inhabited by exactly the zero
    /// value. Terminal, not empty.
    Zero,
}

impl Tangent for Sort {
    #[allow(non_snake_case)]
    fn T(self) -> Self {
        match self {
            Sort::Real { shape, rank } => Sort::Real { shape, rank: rank + 1 },
            Sort::Bool(_) => Sort::Zero,
            Sort::Zero => Sort::Zero,
        }
    }
}
```

- `Real → Real` with `rank + 1`: the carrier (shape) is unchanged, the
  grade rides where `check` can see it. Ranks above 1 exist; forbidding
  them is the signature's job (no generator mentions rank ≥ 2), not the
  sort former's, which stays total.
- `Bool → Zero` realizes the constant-sort story; with the closure law,
  "booleans don't move during delay" is unrepresentable, not checked.
- `Zero → Zero` makes the trivial tangent a fixed point: the tower
  stabilizes, no spurious iterated tangents.

Open choices: `Zero` is shapeless here (evaluators materialize a scalar
zero, never read by any generator — the closure law guarantees it);
`Zero([usize; 2])` is the pragmatic variant if backends prefer a shaped
zero tensor.

### Constant sorts and the Zero sort

Discrete sorts (Bool, Int, BitVec) have a *trivial* tangent, not a missing
one. `T` stays total, and "discrete values don't change during delay" is a
closure property of the signature:

> For a constant sort, `zero` is the only generator whose codomain mentions
> its tangent.

The trivial tangent sort is named **`Zero`**, after its unique inhabitant;
document that it is *inhabited* — a singleton, not the empty type.

### Tangent is universal in the reactive layer

`reactive` imposes `S: Tangent` at the root, by design: modules talk about
evolution over time, so every variable requires a notion of rate — even a
trivial one. Genuine optionality lives in the `F` slot of `Atom<I, J, F>`:
a canonical trivial signature — `Discrete<S>` or `Zero<S>` (only
generator: `zero` into the `Zero` tangents) — makes discreteness nameable
as a type, composing with hybrid modules with no case analysis.

### Time is not a sort former

`d` changes **what** a value is (different sort); the next-face changes
**when** (same sort, next round). Neither is syntax any more: the roles
carry both distinctions, causality is enforced structurally (declared
waits, await acyclicity, feedback only through the register realized by
`bind`), never through the types of `flow`.

### The temporal bi-bundle: X and d as two tangent structures

`X` is a tangent too — the discrete one. *Change actions / difference
categories* (Alvarez-Picallo–Ong; Alvarez-Picallo–Lemay) are the discrete
counterpart of tangent categories: the bundle is `M × ΔM`, a morphism's
derivative maps changes to changes. Reactive modules instantiate them with
the **overwrite change action**: the change to a variable *is* its next
value, so `ΔM = M` — which is exactly why "X is not a sort former": the
sort-action of the difference structure is trivial, where the tangent's
(`T`, rank + 1) is not. Both clauses are the sort-actions of two bundle
structures, one trivial, one graded.

Consequences, each a semantic reading of an existing construct:

- **A `Var` denotes a point of a bi-bundle** with three coordinates: the
  base `v`, the discrete fiber `X(v)` (sort `S`), the continuous fiber
  `d(v)` (sort `T(S)`). Atom boundaries are monomials over these bundle
  components.
- **An atom is a point and two fields**: init picks a point; update is a
  *difference field* (base ⊗ awaited Δ-fibers → own Δ-fibers); delay is a
  *vector field* (base → own T-fibers) — with the section law `p ∘ X = id`
  holding *by construction*, since each fiber wire is attached to its
  variable. The stochastic reading (update = jump part, delay = drift
  part, init = initial distribution) is this structure seen through a
  jump-diffusion generator.
- **Awaiting `w` is reading `w`'s Δ-fiber** — a coordinate another atom
  writes in the same round. The await DAG is the causal well-foundedness
  order on fiber writes, not a scheduling artifact: fibers are written
  before read within a round, bases carry over from the previous one.
- **`skip` and `zero` are the same map in two bundles** — the zero
  sections (`X(v) = v` is zero change; `d(v) = 0` is zero rate) — and
  `havoc` is an arbitrary point. The structural-generator trio was this
  symmetry all along.

Implementation target this licenses: `Sequential` and `Differential`
unify into two instances of *one* parametric bundle-signature notion
(source signature, target universe, sort map, zero section) — the
Δ-instance with target = source and identity sort map, the T-instance
with the tangent target. The two instances are symmetric in their reads
as well: update awaits foreign Δ-fibers, delay awaits foreign T-fibers
(derived observers, e.g. the Lie derivative) — both under the same
acyclicity condition, with cyclic fiber-awaits (algebraic loops, implicit
DAEs) rejected in both bundles.

### Implementation sketch: the bundle-unified atom

The post-bind design, with the bi-bundle analogy as its organizing
principle — one `Bundle` notion, two instances, atoms as a point and two
fields. Toy-scale but every load-bearing decision present.

```rust
/// A bundle signature: the vocabulary for writing *fields* into a bundle
/// over a source universe. ONE notion; the discrete and continuous
/// temporal structures are its two instances.
pub trait Bundle: Signature {
    type Source: Clone + Eq + Debug;
    /// how a base value appears from the fiber universe (reads)
    fn embed(s: Self::Source) -> Self::Sort;
    /// the fiber over a base sort (writes)
    fn fiber(s: Self::Source) -> Self::Sort;
    /// the zero section: the field that leaves the variable unchanged
    fn zero(range: Self::Sort) -> Self;
}

// ---- instance 1: Δ — the overwrite change action (discrete) ----
// ΔM = M: the change IS the next value; target = source, both maps are
// the identity. This IS "X is not a sort former", as an impl.
impl Bundle for Step {
    type Source = Val;                          // Sort = Val too
    fn embed(s: Val) -> Val { s }
    fn fiber(s: Val) -> Val { s }
    fn zero(_: Val) -> Self { Step::Skip }      // zero change: X(v) = v
}

// ---- instance 2: T — the tangent structure (continuous) ----
// The fiber universe is genuinely different: T grades reals and
// collapses constant sorts to the singleton.
pub enum Tan { Val(Val), TReal([usize; 2]), Zero }
impl Bundle for Flow {
    type Source = Val;                          // Sort = Tan
    fn embed(s: Val) -> Tan { Tan::Val(s) }
    fn fiber(s: Val) -> Tan {
        match s { Val::Real(sh) => Tan::TReal(sh), Val::Bool => Tan::Zero }
    }
    fn zero(_: Tan) -> Self { Flow::ZeroGrad }  // zero rate: d(v) = 0
}

/// A field into bundle B: reads bases and already-written fibers, writes
/// its own fibers. This ONE shape is init, update, and delay.
pub struct Field<B: Bundle> {
    op: B,
    writes: Vec<VarId>,       // own fibers
    reads_base: Vec<VarId>,   // latched values
    reads_fiber: Vec<VarId>,  // the awaits: Δ-fibers (update), T-fibers (delay).
}

/// One checker for every role — faces are (bundle, list), nothing else.
fn check_field<B: Bundle>(vars: &[Var], f: &Field<B>) -> Result<(), String> {
    let reads = f.reads_base.map(B::embed ∘ sort)
        .chain(f.reads_fiber.map(B::fiber ∘ sort));
    let writes = f.writes.map(B::fiber ∘ sort);
    f.op.check(&reads, &writes)
}

/// The zero-section synthesizer — today's Block::zero AND Block::skip,
/// as one generic function. (The zero section reads its base exactly when
/// the fiber universe contains it: skip reads v, zerograd reads nothing.)
fn zero_field<B: Bundle>(vars: &[Var], ctrl: &[VarId]) -> Vec<Field<B>> { … }

/// An atom: a point and two fields, one per bundle.
pub struct Atom<D: Bundle<Source = Val>, C: Bundle<Source = Val>> {
    init: Vec<Field<D>>,   // a point: fields with no base reads
    step: Vec<Field<D>>,   // difference field  (Δ)
    flow: Vec<Field<C>>,   // vector field      (T)
}
```

The atom kinds become definitions, not constructors: a **sequential**
atom has its dynamics in Δ and `zero_field::<Flow>` as its flow; a
**differential** atom has its dynamics in T and `zero_field::<Step>` as
its step; a **combinatorial** atom is `init == step` with zero flow; a
**hybrid** atom is both fields nontrivial — no new machinery, which is
the strongest evidence the analogy carries weight.

What the unification buys, concretely:

- `Sequential`/`Differential` collapse into `Bundle` at Δ and T; `skip`
  and `zero` are the same associated function; `Block::skip` and
  `Block::zero` (two synthesizers in today's `atom.rs`) become one
  generic `zero_field`.
- `check_field` is the entire static semantics; init is the degenerate
  field with no base reads (a point); faces never exist as data.
- Awaits are the Δ-instance's `reads_fiber`; the same slot in the
  T-instance is the DAE question, demoted from structural exception to
  per-instance policy. One causality checker serves both bundles:
  topologically order fiber-writes before fiber-reads.
- `bind` keeps one demand-driven wire table per bundle instance, same
  fold as before.

## Await: reading the fiber

Awaiting is not a scheduling annotation — it is *which coordinate of the
bi-bundle an atom reads*. Reading `x` reads the **base**: the value
carried over from the previous round, always available, never
order-constraining. Awaiting `x` reads the **Δ-fiber** `X(x)`: a
coordinate that `x`'s controller writes *in the same round*, so the read
is well-defined only after that write. The await relation is exactly the
fiber-read dependency graph; the linear order atoms must respect is its
topological order; a cyclic await is a causal cycle on fibers and is
rejected. Nothing else about atom order matters — base reads commute.

Examples, in the surface syntax (target vocabulary: `Op` — today's
python surface still spells it `Term`):

- **Registered vs. wire-like coupling** (Moore vs. Mealy). The sequential
  atom reads the base — output lags input by a round, no ordering
  constraint:

  ```python
  Op(LRA.Id(), [X(y)], [x])      # y' = x   (through the register)
  ```

  The combinatorial atom awaits — output follows input within the round,
  and the atom must run after x's controller:

  ```python
  Op(LRA.Id(), [X(y)], [X(x)])   # y' = x'  (a wire, zero delay)
  ```

- **Causal cycles are fiber cycles.** Cross-coupled swap through the
  registers is fine — two base reads, no ordering at all:

  ```python
  Op(LRA.Id(), [X(x)], [y]);  Op(LRA.Id(), [X(y)], [x])
  ```

  The same shape on fibers is a zero-delay loop and is rejected (see
  `cannot_compose_example_tiny1_with_cyclic_await`):

  ```python
  Op(LRA.Id(), [X(x)], [X(y)]);  Op(LRA.Id(), [X(y)], [X(x)])
  ```

- **Init reads are fiber reads by necessity.** At time zero no base
  exists — nothing was carried over — so an init block can only await:
  initialization cascades (`X(y) = f(X(x0))`) are Δ-fiber chains, ordered
  by the same acyclicity condition as updates.

- **The continuous mirror, and its first client.** A delay net reading a
  base is a vector field (`d(p) = v`: chained integrators — no ordering
  constraint, integration decouples them). A delay net reading a foreign
  **T-fiber** would be the continuous await:

  ```python
  Op(LRA.Linear(A, b), [d(y)], [d(x)])   # dy = A·dx : algebraic coupling
  ```

  This is delay's wait list — the design admits *acyclic* T-fiber reads,
  with the **Lie-derivative observer** as the motivating client:
  `d(v_V) = ∇V(x) · f(x)` reads the plant's `d(x)` wires, an awaited
  derivative (observer depends on plant, never conversely). The same
  causality condition as Δ-awaits applies; genuinely implicit DAEs
  (cyclic derivative dependencies, algebraic loops) stay out.

The symmetry, in one table:

|                      | Δ (discrete)         | T (continuous)               |
|----------------------|----------------------|------------------------------|
| base read            | `x` — previous round | `x` — current state          |
| fiber read           | `X(x)` — **await**   | `d(x)` — **await** (observer)|
| ordering constraint  | await acyclicity     | same, if the gate opens      |
| cycle                | zero-delay loop      | algebraic loop               |

## Naming record

Settled: **`Op`** replaces `Term` (and the interim `Box`); the composite
structure needs no separate noun — a net/tape *is* a composed op, and
`Op<G>: Signature` makes that a type-level fact. **`flow`** is the crate:
data and control flow along wires; the hybrid-systems pun ("flow" also
names continuous evolution, flow vs. jump) is accepted — the crate name
lives in paths, the dynamics term in prose. **`Wire`** is reclaimed as the
bind-level coordinate.

Considered and rejected, with reasons:

- **`Box`**: shadows the prelude's `std::boxed::Box`; visualization-first.
- **`Fun`**: havoc and uninterpreted generators make ops relations, not
  functions; one letter from the prelude trait `Fn`.
- **`Diagram`**: names the picture, not the structure — kept in prose for
  semantics ("ops are string diagrams / tapes / open hypergraphs").
- **`Circuit`**: right for the discrete fragment, wrong connotation for a
  delay net (a system of ODEs is not a circuit).
- **`Net`**: served as the working name (netlists + neural nets, both
  literal here); absorbed into `Op` when the composite and the atomic
  unified. Disambiguation to keep in prose: dataflow networks, not Petri
  nets.
- **`Kernel`**: honest for the update role (Markov kernel) but
  role-specific — an init is an initial distribution — and overloaded.
- **`Gen`** (for the composite): the generator names a *role* (the thing
  inducing evolution — the atom, in jump-diffusion terms: update = jump
  part, delay = drift part, init = initial distribution), not the syntax;
  and it collides with the signature-symbol vocabulary.
- **`Port`/`Aspect`** (variable faces): made redundant by roles — the face
  of every position is a function of (role, list).

## Open items

- `Module` as a `Signature` (modules as ops in larger diagrams — the
  wiring-operad picture): definable via the sorted boundary, but the
  forgetful map drops visibility and await metadata; a deliberate step.
- ~~Delay reading derivatives~~ settled: delay carries a wait list of
  awaited derivatives (see "Await: reading the fiber") — acyclic T-fiber
  reads admit derived observers (the Lie derivative's client) under the
  same causality condition as awaits; algebraic loops stay out.
- LRA lacks the time form `dt` and the degree-conserving flow generator
  `Drift(A, b)` (`d(y) = (A·x + b)·dt`, reading a base and `dt`): only
  `zero` writes 1-forms today, so no nontrivial flow is expressible —
  prerequisite for every continuous example and, via the same graded
  product (`∇V(x) · dx`), for the Lie-derivative observer. Decide where
  `dt` lives: a distinguished per-module 1-form minted by `bind`, or the
  derivative face of an explicit clock variable.
- Lifting `havoc`/`skip`/`zero`/`assume` from generators to composite ops.
- `Op` constructor ergonomics (it is the most-typed name in the API).
- Store branding for `OpId` (cross-store misuse).
- When the `Zero` sort lands, document it as inhabited.

## Potential connections

Recorded as leads, not claims:

- **Term graphs / gs-monoidal categories** (Corradini–Gadducci):
  single-writer/multi-reader dataflow, explicit copy and discard, copy not
  natural (HAVOC). The formal home for the circuit fragment.
- **Tape diagrams** (Bonchi–Di Giorgio–Santamaria): string diagrams for rig
  categories with finite biproducts — ⊗ for dataflow, ⊕ for control flow,
  sums-of-products normal form. The calculus `flow::Op` implements;
  guarded commands are its normal form, `Ite` its distributor.
- **Open hypergraphs** (Wilson–Zanasi; the `open-hypergraphs` crate):
  the free-SMC-arrow datastructure for the circuit fragment. Reuse plan: a
  functor from `flow` (visualization, rewriting, optic-based autodiff) —
  not a replacement: their equations include special Frobenius (merging,
  illegal here) and their nodes are local indices.
- **Markov categories** (Fritz): copy/discard without merging —
  structurally our discipline; morphisms are kernels. Stochastic vocabulary
  maps onto atoms: update = (Markov) kernel, delay = infinitesimal
  generator (jump diffusions unify both), init = initial distribution.
- **RVSDG / MLIR regions** (compilers): folded control-flow nodes inside
  dataflow with explicit expansion passes — engineering precedent for
  lazy structure and on-demand normal forms.
- **Monoidal streams** (Di Lavore–de Felice–Román): candidate denotational
  semantics for the reactive layer — an atom as the state-machine
  presentation of a stream, `hide` matching delayed feedback `fbk`.
  Unproven; sequential fragment only.
- **`Span(Graph)` / circuit algebras** (Katis–Sabadini–Walters): parallel
  composition with coupling and a hiding operator matching ours.
- **Operads of wiring diagrams / open dynamical systems** (Spivak;
  Vagner–Spivak–Lerman): composition-as-wiring; the natural home for
  modules-as-ops and the hybrid (delay) dimension.
- **Tangent categories** (Rosický, Cockett–Cruttwell): the sort-level `T`
  as tangent functor; discrete objects = trivial tangents.
- **Change actions / difference categories** (Alvarez-Picallo–Ong,
  Alvarez-Picallo–Lemay): the discrete tangent structure behind `X` — see
  "The temporal bi-bundle"; overwrite change action `ΔM = M`.

## The stack

```
theory     Signature, Sort, Tangent,   — generators, sorts, arity checking;
           LRA, LIA, BV                  Theory: Signature reserved for axioms
flow       Op<G: Signature>            — free rig construction: dataflow ⊗,
                                         control flow ⊕, lazy, arena/OpId
reactive   Var, Atom, Module           — names, time roles, awaits, visibility
bind       bind: Module → Instance     — wires, coupling, Binding<T>, SSA
python       bindings, sugar, gym, torch, smt — clients of bind
```

An op is generators composed over positions; a signature provides the
generators and sorts; an atom is three signature elements bound to
variables under temporal roles; a module is atoms in await order behind an
interface; binding makes it all one wired instance.
