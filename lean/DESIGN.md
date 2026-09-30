theory — what operations exist
 - MultiSort (a set of sorts a value of a theory can have)
 - Signature: a type: List MultiSort \to List MultiSort
 - Transition: class that everything that has signature implements
 - Theory over MultiSorts
    - and generators (the primitive operation symbols). Each generator has its own signature.

 Tangent extends sorts with a differential grade, so derivatives are typed. LRA, LIA, BV are the built-in signatures. Everything else depends on this crate and nothing here depends on anything.

dataflow — computation, without time
How generators compose into larger computations: Op, the free construction over any signature — sequential and parallel composition, plus branching, all as data flowing through a graph. There are no variables, no wires, and no notion of time here; identity is positional. Since Op itself implements Signature, composites nest: an op built from generators can be used wherever a generator is expected. (note that Op can be implemented differently, as a tape diagram, an SSA, a guarded command, and there can be multiple implementations)

reactive — state, over time
What has a name and how it evolves: Var (a sort-carrying name), Atom (three fields binding computations to variables — init sets a variable, next steps it each round, flow moves it continuously), and Module (atoms glued by await order and visibility). Purely syntactic — all checking is sort checking plus ordering discipline — and it depends only on theory: it never sees dataflow.

bind — where the two meet
The only crate that knows both worlds. bind elaborates a module into its bound form, minting wires: shared variables couple automatically (same variable, same wire), internals stay fresh by construction. Wires exist only here — the syntax crates cannot even name one — and every semantic consumer (evaluation, SMT encoding, torch execution, display, monitors) is a client of bind.

The rules that keep it clean
dataflow and reactive are mutually ignorant; they share only theory and meet only in bind.
Three identities, one per crate that names things: Op (dataflow), Var (reactive), Wire (bind). Nothing else mints an identifier.
"Wires exist only below bind" is enforced by the crate graph, not by convention.
In one sentence: theory says what operations exist, dataflow composes them into computations, reactive attaches them to named state evolving over time, and bind wires it all together for the backends.
