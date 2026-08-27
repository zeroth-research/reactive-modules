# Design: diagrams, signatures, and the differential structure

Proposal for a codebase splits of the base into a wire-diagrammatic substrate and a reactive
layer on top of it.

## Crates

- **`diagram`** — the dataflow substrate: `Wire`, `Box`, `Diagram`. No
  notion of rounds, state, or time. Named after its headline type
  (`regex::Regex` idiom); `diagram::Box` self-qualifies past the
  `std::boxed::Box` clash.
- **`reactive`** — the reactive-modules formalism: `Var`, `Atom`, `Module`.
  Everything temporal lives here: latched/next/derivative views, rounds,
  await ordering, composition, hiding. The split is *timeless vs. temporal*.

## Terminology

A **`Diagram`** is an assembly of **`Box`es** connected by **`Wire`s**; its
derived read/write lists are its **boundary** (`dom`/`cod`). The names
replace the old vocabulary: **`Box` replaces `Term`** and **`Diagram`
replaces `Block`**, and block becomes a first-class citizen (`Wire` was already the literature term). Wires have one
writer and any number of readers; construction rejects double writes and
reads-before-writes. The diagram vocabulary stops at the substrate:
`Var`/`Atom`/`Module` are not diagram notions — an atom *holds* three
`Diagram`s.

## Signatures

The trait **`Signature`** is a many-sorted signature: `Signature::Sort` is
the sort set, the implementing enum's variants are the operation symbols
(**generators**), `check` is the arity discipline. `LRA`, `LIA`, `BV` are
signatures.

The accessors follow the vocabulary: a wire's `dtype` becomes **`sort()`**
(the wire's sort), and a box's `itype` becomes **`gen()`** (the generator
the box applies).

The name **`Theory`** is reserved for the upcoming subtrait
`Theory: Signature` carrying axioms (equations between `Diagram`s).
Rationale: arity checking runs at construction time everywhere; axioms feed
rewriting/equivalence machinery; uninterpreted signatures need no
empty-axioms boilerplate.

**`Sort`** is both the algebraic partner of "signature" and SMT-LIB
vocabulary (`Real`, `Int`, `(_ BitVec n)`), so both audiences read it
natively.

## The differential structure

### Today: degree as a grading

The wire `degree` (`0`-form = value, `1`-form = derivative) lives on
`Wire`. It is imposed by the reactive formalism — every `Var` mints a
derivative wire before any signature is in sight — not by particular
theories. The `u8` encoding assumes `d(S)` has the same carrier as `S`,
which holds for vector-space-like sorts (LRA). **Migration trigger**: the
first sort with `d(S) ≠ S` (units, manifolds).

### Then: sort-level tangents

```rust
/// On the sort type: sorts closed under the tangent former.
pub trait Tangent: Sized {
    #[allow(non_snake_case)]
    fn T(&self) -> Self;
}

/// On the signature: the generators over tangent sorts.
pub trait Differential: Signature
where
    Self::Sort: Tangent,
{
    fn zero(range: &Self::Sort) -> Self;
}
```

The former lives on the *sort* because `Var::new` needs it with no
signature at hand (a module carries three signatures sharing one sort
type).

**`Tangent` removes `degree` from `Wire`**: the differential form becomes
the `Sort`'s responsibility. A wire is just an identity with a sort; a
1-form wire is one whose sort is in the image of `T`. `Var::new` mints the
derivative wire as `Wire::new(s.T())` instead of tagging a degree, `check`
consumes plain sorts instead of `(sort, degree)` pairs, and "derivative of
an integer wire" becomes unwritable rather than merely useless.

The method is `T` (tangent-functor notation, like `X`/`d`); note it
coexists with the generic-parameter convention `T` — different namespaces,
but nearby generics deserve care.

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
    fn T(&self) -> Self {
        match *self {
            Sort::Real { shape, rank } => Sort::Real { shape, rank: rank + 1 },
            Sort::Bool(_) => Sort::Zero,
            Sort::Zero => Sort::Zero,
        }
    }
}
```

- `Real → Real` with `rank + 1` is the old wire `degree` absorbed into the
  sort: the carrier (shape) is unchanged, the grade rides where `check` can
  see it. Ranks above 1 now exist; forbidding them is the signature's job
  (no generator mentions rank ≥ 2), not the sort former's, which stays
  total.
- `Bool → Zero` realizes the constant-sort story; with the closure law,
  "booleans don't move during delay" is unrepresentable, not checked.
- `Zero → Zero` makes the trivial tangent a fixed point: the tower
  stabilizes at the first step, no spurious iterated tangents.

Open choices: `Zero` is shapeless here (the terminal object is unique;
evaluators materialize a scalar zero, never read by any generator — the
closure law guarantees it); `Zero([usize; 2])` is the pragmatic variant if
backends prefer a shaped zero tensor. `rank` lives only on `Real`, since
`Bool`'s tower collapses immediately. Sort equality now distinguishes
ranks, so rank-generic generators (`Add` over any grade) check "same shape,
same rank" — the same comparison the `(sort, degree)` pairs run today,
relocated.

### X is not a sort former

`d` changes **what** a value is (different sort); `X` changes **when** it
is (same sort, next round). Diagrams apply generators across latched and
next wires freely, so the substrate stays time-free and `X(v)` has the sort
of `v`. Causality is enforced structurally (declared waits, await
acyclicity, feedback only through the register), not through the types.

### Constant sorts and the Zero sort

Discrete sorts (Bool, Int, BitVec) have a *trivial* tangent, not a missing
one. `T` stays total, every `Var` keeps a derivative wire, and "discrete
values don't change during delay" is a closure property of the signature:

> For a constant sort, `zero` is the only generator whose codomain mentions
> its tangent.

The trivial tangent sort is named **`Zero`**, after its unique inhabitant;
the only generator writing a `Zero` wire is `zero`. Document that `Zero` is
*inhabited* — a singleton, not the empty type.

### Tangent is universal in the reactive layer

`reactive` imposes `S: Tangent` at the root, by design: modules talk about
evolution over time, so every module variable requires a notion of rate —
even a trivial one. The bound is one line to satisfy (`T(s) = Zero` for
discrete sorts), and uniformity is required by the domain: hybrid modules
mix discrete and continuous variables in one module, so bifurcating `Var`
or `Module` by tangent availability would fight the formalism.

Genuine optionality lives in the `F: Differential` parameter of
`Module<I, J, F>`: a canonical trivial signature — named `Discrete<S>` or
`Zero<S>` (only generator: `zero` into the `Zero` tangents) — makes
discreteness nameable as a type. `Module<I, J, Discrete<S>>` composes with
hybrid modules with no case analysis — it *is* a hybrid module whose
continuous dynamics are trivial.

## Potential connections

Recorded as leads, not claims:

- **Term graphs / gs-monoidal categories** (Corradini–Gadducci): diagrams
  with single-writer/multi-reader wires, implicit copy and discard, copy
  not natural (HAVOC). The closest formal home for the substrate.
- **Monoidal streams** (Di Lavore–de Felice–Román): candidate denotational
  semantics for the reactive layer — an atom is the state-machine
  presentation of a stream (memory = latched state, "now" = update block),
  and `hide` matches the delayed-feedback operator `fbk`. Unproven; would
  cover the sequential fragment only.
- **`Span(Graph)` / circuit algebras** (Katis–Sabadini–Walters): parallel
  composition with coupling and a hiding operator matching ours.
- **Operads of wiring diagrams / open dynamical systems** (Spivak;
  Vagner–Spivak–Lerman): composition-as-wiring; the natural home for the
  hybrid (delay) dimension.
- **Tangent categories** (Rosický, Cockett–Cruttwell): the sort-level `T`
  as tangent functor; discrete objects = trivial tangents.

## The stack

```
theory     Signature, Sort, Tangent,   — generators, sorts, arity checking;
           LRA, LIA, BV                  Theory: Signature reserved for axioms
diagram    Wire, Box, Diagram          — typed string diagrams
reactive   Var, Atom, Module           — time, feedback, visibility, composition
python       python bindings, sugar, gym, torch, smt
```

A diagram is boxes over wires; a signature provides the generators and
sorts; a reactive module is diagrams arranged in time behind an interface.