# Design note: diagrams, signatures, and the differential structure

*A record of the naming and architecture decisions around splitting `base`
into a string-diagrammatic substrate and a reactive layer (August 2026).*

## The split and the crate names

`Wire` and `Term`/`Block` move out of `base` into a crate dedicated to
representing the dataflow substrate. The remaining layer (`Var`, `Atom`,
`Module`) is the reactive-modules formalism proper. Names:

- **`diagram`** — the substrate crate. Considered against `circuit` (best
  cross-community readability: arithmetic circuits for theorists, gates and
  wires for hardware people) and `netlist` (most precise EDA term, but
  jargon). `diagram` won once the string-diagrammatic treatment became a
  planned deliverable rather than an evocation: the crate then delivers
  literally what the name promises. The `regex::Regex` / `url::Url` idiom
  makes `diagram::Diagram` natural.
- **`reactive`** — the remaining layer. The split is precisely *timeless vs.
  temporal*: `diagram` has no notion of rounds, state, or time; `reactive`
  adds everything that makes it react — latched/next/derivative views,
  rounds, await ordering, composition, hiding. `modules` was the runner-up,
  rejected for the "the `modules` crate's `module` module's `Module`"
  friction with Rust's own vocabulary.

## String-diagrammatic terminology

The purist vocabulary was adopted for the substrate: **`Wire`**, **`Box`**,
**`Diagram`** (Selinger's survey: diagrams consist of wires and boxes — so
`Wire` was already the literature term). The `std::boxed::Box` clash is
accepted; `diagram::Box` reads as self-qualifying. A diagram's derived
read/write lists are its **boundary** (`dom`/`cod`, or `inputs`/`outputs`).

The formal backing, layer by layer:

- **Substrate**: blocks with single-writer/multi-reader wires are term
  graphs, the arrows of **gs-monoidal categories** (Corradini–Gadducci) —
  implicit copy (fan-out) and discard (unused writes), no wire merging (not
  Frobenius), copy not natural (HAVOC). The construction invariants
  (`try_from_iter`: no double write, no read-before-write) are that
  presentation's well-formedness conditions.
- **Reactive layer**: latched/next is **delayed feedback** through a
  register; the await-acyclicity check is the guardedness condition. This is
  the Katis–Sabadini–Walters circuit-algebra / `Span(Graph)` world (their
  hiding operator matches ours exactly), with modern semantics in monoidal
  streams (Di Lavore–de Felice–Román). Composition-as-wiring matches
  Spivak's operad of wiring diagrams; the hybrid part matches algebras of
  open dynamical systems (Vagner–Spivak–Lerman).
- The vocabulary stops at the substrate: `Var`/`Atom`/`Module` are not
  diagram notions; the correct amount of leakage is "an atom holds three
  `Diagram`s".

## Theory → Signature

The trait `Theory` is renamed **`Signature`**: a many-sorted signature is a
set of sorts plus operation symbols with arities (Goguen–Meseguer), which is
exactly the trait — `Signature::Sort` is the sort set, the implementing
enum's variants are the symbols, `check` is the arity discipline. The
multi-input/multi-output shape makes it precisely a **monoidal signature**
(tensor scheme). The op enums (`LRA`, `LIA`, `BV`) are signatures; their ops
are **generators** (the `itype` of a box).

`Theory` over-promised: a theory is a signature *plus axioms*, and axioms
are planned but not present. The name `Theory` is reserved for the future
subtrait `Theory: Signature` carrying equations (pairs of `Diagram`s over
the signature asserted equal — a monoidal theory presentation). Splitting by
consumer: arity checking is construction-time and everywhere; axioms feed
rewriting/equivalence machinery. Uninterpreted signatures are honestly
signatures — no empty-axioms boilerplate.

**`Sort`** stays: it is the textbook partner of "signature" (a signature
*is* sorts + symbols) and simultaneously SMT-LIB vocabulary (`Real`, `Int`,
`(_ BitVec n)` are sorts there), bridging both audiences. The PROP-dialect
alternative ("colors") is niche; `Type` and `Object` are unusable in code.

## The differential structure

### Degree is a grading, until it isn't

The wire `degree` (`0`-form = value, `1`-form = derivative) currently lives
on `Wire`, and correctly so: it is a grading imposed by the *reactive
formalism* (every `Var` mints a derivative wire before any signature is in
sight; delay blocks write 1-forms uniformly), not by particular theories.
The category's objects are graded sorts; signatures constrain generators
over them.

The flat `u8` encoding assumes `d(S)` has the same carrier as `S` — true
for vector-space-like sorts (why LRA works), false the moment sorts with
units, manifolds, or non-differentiable carriers arrive. **Trigger for
migration**: the first sort with `d(S) ≠ S`.

### Sort-level tangents

The migrated design splits the differential structure by who needs it:

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

The former lives on the *sort* because `Var::new` needs it with no signature
at hand (a module carries three signatures sharing one sort type; only one
is differential). With sort-level tangents, `degree` disappears from `Wire`
(a 1-form wire is a wire whose sort is in the image of `T`), `check`
consumes plain sorts, and "derivative of an integer wire" becomes
unwritable rather than merely useless. Backing: **tangent categories**
(Rosický, Cockett–Cruttwell) for the endofunctor-on-objects; *differential
signature* as the natural coinage in the "differential ring/field = ring
with derivation" convention; the de Rham complex for the grading vocabulary
(0-forms, 1-forms, `covector`) already in use.

The method is `T`, matching the tangent-functor notation `TM` and the
`X`/`d` precedent of notation-faithful names (`#[allow(non_snake_case)]`,
as for those). Accepted trade-offs, noted for the record: `T` is also
Rust's conventional generic type parameter (the two live in different
namespaces, so `T(s)` resolves fine even inside `fn foo<T>`, but nearby
generics deserve care), and the name takes the tangent side of the
tangent/cotangent reading — `d`/`covector` speak de Rham while
delay-writes-velocities speaks tangent bundle; isomorphic for
finite-dimensional vector spaces, to be revisited only if sorts where they
differ ever arrive.

### X is not a sort former

`d` changes **what** a value is (a rate; different sort). `X` changes
**when** it is (same carrier, next round) — and blocks apply generators
across latched and next wires freely, so the substrate is time-free and
`X(v)` has the sort of `v`. The mathematics for temporal typing exists (the
next/later modality ▷, guarded recursion, LTL-as-types, Lustre's clock
calculus) but exists to enforce causality *through types* — and causality
is already enforced *structurally* (atoms declare waits; await acyclicity
is checked; feedback only through the register). Slogan: `d` is a modality
on values, `X` a modality on time; values live in the sorts, time in the
structure.

### Constant sorts and the Zero sort

Sorts without genuine tangents (Bool, Int, BitVec) are not
non-differentiable — their tangent is *trivial*. Backing: discrete spaces
have zero tangent bundles (in tangent categories, "discrete objects" are
those where the zero section is iso); in differential algebra the elements
with `d(c) = 0` are the **constants**; hybrid-systems practice gives
discrete state the implicit flow `ẋ = 0` (which `Block::zero` synthesis and
`BoolZerograd` already implement). So `tangent` stays total, every `Var`
keeps a derivative wire, and the invariant "discrete values don't change
during delay" becomes a closure property of the signature:

> For a constant sort, `zero` is the only generator whose codomain mentions
> its tangent.

The trivial tangent sort is named **`Zero`**, not `Unit`: the singleton is
named after its unique inhabitant (standard for zero spaces; "zero tangent
space" / "zero bundle" are the literature's own names), the generator
correspondence is mnemonic (the only generator writing a `Zero` wire is
`zero`), and `Unit` actively collides with differential geometry's "unit
tangent" (norm-1 vectors, the unit tangent bundle). One doc obligation:
state that `Zero` is *inhabited* — the terminal object of the sort
category, not the empty type.

### Tangent is universal in the reactive layer

`Var::new` mints the derivative wire, so `reactive` imposes `S: Tangent` at
the root — the trait is effectively non-optional. This is accepted, and
argued for rather than worked around: **modules talk about evolution over
time, so every module variable requires a notion of rate — even if that
rate is trivial**. A discrete variable's rate is not missing; it is the
statement "this cannot move during delay", i.e. a tangent wire of sort
`Zero`. The bound has the same moral status as the `Eq + Clone + Debug`
bounds the layer already imposes: universal, and one line to satisfy
(`T(s) = Zero`-of-shape for discrete sorts).

Uniformity is required by the domain, not just convenient: hybrid modules
mix discrete and continuous variables in one module (a timed automaton's
location and clock), so bifurcating `Var` or `Module` by tangent
availability would fight the formalism. Tangent categories agree: discrete
objects form a full subcategory of the tangent category — same universe,
trivial tangent — not a separate world. A module never changes nature; some
of its tangents are just `Zero`.

Where genuine optionality lives instead:

- **The `F: Differential` type parameter of `Module<I, J, F>`** is where
  "is differentiation available?" is asked, per module, at compile time. A
  canonical trivial signature — `Discrete<S>`, whose only generator is
  `zero` into the `Zero` tangents — makes discreteness *nameable as a
  type*: `Module<I, J, Discrete<S>>` is the discrete module, its
  synthesized delay blocks are all-zeros (as today), and it composes with
  hybrid modules with no case analysis, because it *is* a hybrid module
  whose continuous dynamics happen to be trivial. Change of nature is a
  change of type parameter connected by an embedding, not a fork in the
  data structures.
- **Bounded inherent impls** (`impl<S: Tangent> Var<S> { fn d(...) }`) are
  the reserve mechanism if sorts refusing the trait are ever wanted: the
  tangent-touching API exists only under the bound. Held in reserve, not
  led with — once `Var` construction is conditional, every generic
  container inherits the split, maintaining two worlds to make impossible
  what the `Zero` design already makes harmless.

## Semantics: monoidal streams

The deepest connection in the citation list: **monoidal streams**
(Di Lavore–de Felice–Román, LICS 2022) are a strong candidate for the
*denotational semantics* of the reactive layer, not merely an analogy.
`Stream(C)` — causal stream transformers over any symmetric monoidal
category `C`, built coinductively from a memory object and a "now" morphism
`M ⊗ X → M ⊗ Y` — is the canonical way to add guarded feedback to a base
category, and the stack factors exactly that way:

- **`reactive ≈ Stream(diagram(Σ))`**: the diagram crate over a signature is
  the base category; the reactive layer is its stream category.
- **An atom is the state-machine presentation of a stream**: memory = the
  controlled (latched) variables, "now" = the update block
  (`M ⊗ X → M ⊗ Y`), initial memory = the init block.
- **`hide` is `fbk`**: the feedback operator internalizes a wire — output
  fed back with a one-round delay, removed from the boundary — which is
  precisely hiding a controlled variable into latched memory. This is the
  modern formulation of the KSW-hiding match noted above.
- **Await acyclicity is causality**: `fbk` is guarded (feedback only through
  the delay); the topological-sort check discharges syntactically what the
  coinduction requires semantically.
- **Nondeterminism fits**: the construction works over copy-discard
  categories where copy is not natural — exactly the gs-monoidal property
  of the base (HAVOC). Their Markov-category instantiation marks a future
  direction: stochastic reactive modules with unchanged syntax.
- **Intensional vs. extensional**: modules are intensional presentations;
  streams are behaviors up to coinductive (observational) equivalence. The
  semantics functor `Module → Stream(diagram(Σ))` is what the future
  `Theory` axioms would quotient toward.

Caveat, recorded honestly: this is a correspondence to be proven, not
cited — the statement covers the sequential fragment; the hybrid delay
dimension has no counterpart in discrete-time streams and stays with the
open-dynamical-systems operad story.

## The stack

```
diagram    Wire, Box, Diagram          — typed string diagrams (gs-monoidal)
theory     Signature, Sort, Tangent,   — generators, sorts, arity checking;
           LRA, LIA, BV                  Theory: Signature reserved for axioms
reactive   Var, Atom, Module           — time, feedback, visibility, composition
zrth       python bindings, sugar, gym, torch, smt
```

Each crate is one sentence: a diagram is boxes over wires; a signature
provides the generators and sorts; a reactive module is diagrams arranged
in time behind an interface.
