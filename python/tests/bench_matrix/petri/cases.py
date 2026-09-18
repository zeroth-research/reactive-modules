"""Every property the `petri` suite asks, and what is known about it.

A row names either a net in `nets.py` (`net=`, built by
`../petri_mod.py`) or one of the two modules written out in this directory
(`mod=`), which carry clocks and a fluid level rather than only a marking.

The property is over `s0..sN-1`, which for a net is its `places` tuple in
order -- so `mutex`'s `(<= (+ s2 s5) 1)` is `c1 + c2 <= 1`, and a `-rr`
variant's counter is the one component after the last place. The selector's
`--pre` is not written here: it follows from the net's transition count, and
`suites.py` derives it.

`truth` decides whether a route's answer is *correct* -- only a row whose
property fails may honestly be `REFUTED` -- and every value is checked by
`tests/bench_matrix/test_rows.py`, which fires the net and looks at the
markings it reaches.

Three kinds of row are worth telling apart:

  **P-semiflow rows** (`semiflow`, `slots`, `permits`, `forks`) are linear
  equalities over the marking, inductive by construction. Like the fbk
  suite's `inv-` probes they measure the plumbing rather than the search --
  but they are also the invariant every *other* row's proof goes through, so
  a route that cannot take one cannot prove the rest either.

  **Extension rows** (`inhibit/queue-bound`, `reset/slots-eq`,
  `transfer/conserved`, `sem-cont/integral`) are each about an arc a
  classical net does not have. Two hold *because* of the extension and two
  are classical invariants the extension breaks, so the four together say
  what each arc costs rather than only that it exists.

  **Recurrence rows** live on the `-rr` nets. Under a free input `G F p` is
  false for nearly every `p` -- the input may select a dead transition for
  ever -- so recurrence is asked of the round-robin variant, where the
  scheduler is part of the state.
"""

C = dict

CASES = [
    # ───────────────────── mutual exclusion ───────────────────────
    C(net="mutex", name="mutex", kind="safety",
      P="(<= (+ s2 s5) 1)", truth="holds",
      note="`c1 + c2 <= 1`: the property the net exists for. It follows from "
           "the semaphore semiflow below and from nothing weaker, so a route "
           "that cannot find `c1 + c2 + sem = 1` cannot prove it either"),
    C(net="mutex", name="semiflow", kind="safety",
      P="(= (+ s2 s5 s6) 1)", truth="holds",
      note="the semaphore's P-semiflow, stated directly. Inductive by "
           "construction -- every transition that takes the token gives it "
           "back -- so this row measures whether a route can carry a linear "
           "equality at all, separately from whether it can discover one"),
    C(net="mutex", name="never-crit", kind="safety",
      P="(= s2 0)", truth="fails",
      note="process 1 never enters its critical section. Refuted by two "
           "firings, `req1` then `enter1` -- the shortest counterexample in "
           "the suite, and a control for whether a route reports `REFUTED` "
           "at all"),
    C(net="mutex-rr", name="enters", kind="buchi",
      P="(= s2 1)", truth="holds",
      note="under round-robin, process 1 enters infinitely often. The net's "
           "orbit is exactly the six-step cycle the scheduler walks, so the "
           "ranking function is the scheduler's counter -- the easiest "
           "recurrence in the suite, and the baseline the harder ones are "
           "read against"),

    # ──────────────────── producer / consumer ─────────────────────
    C(net="prodcons", name="buffer-bound", kind="safety",
      P="(<= s2 3)", truth="holds",
      note="the buffer never overflows. A classical net cannot test a place "
           "for an upper bound, so the bound is carried by the complementary "
           "place and this row is provable only through `slots`"),
    C(net="prodcons", name="slots", kind="safety",
      P="(= (+ s2 s3) 3)", truth="holds",
      note="`buf + free = 3`, the complementary-place semiflow that *is* the "
           "capacity. The same shape as `reset/slots-eq` below, in a net "
           "where it survives"),
    C(net="prodcons", name="tight", kind="safety",
      P="(<= s2 2)", truth="fails",
      note="a buffer bound one slot too small. Refuted, but only after the "
           "producer runs three times without the consumer -- a "
           "counterexample a bounded search has to be deep enough to reach"),
    C(net="prodcons-rr", name="empties", kind="buchi",
      P="(= s2 0)", truth="holds",
      note="under round-robin the buffer is empty infinitely often. The "
           "producer and consumer alternate, so the buffer never holds more "
           "than one item -- a *reachability* fact the recurrence needs, and "
           "one that is false of the free-input net"),

    # ───────────────────── readers / writers ──────────────────────
    C(net="readwrite", name="exclusion", kind="safety",
      P="(not (and (>= s1 1) (>= s2 1)))", truth="holds",
      note="no reader while a writer holds the resource. The protocol is an "
           "arc *weight* -- a writer takes all three permits -- so this is "
           "the suite's one row where the invariant has to reason about "
           "weighted arcs rather than unit ones"),
    C(net="readwrite", name="permits", kind="safety",
      P="(= (+ s1 (* 3 s2) s3) 3)", truth="holds",
      note="`reading + 3 writing + sem = 3`: the weighted P-semiflow. The "
           "coefficient is what makes it different in kind from `mutex`'s, "
           "and what a route restricted to unit coefficients cannot state"),
    C(net="readwrite", name="one-reader", kind="safety",
      P="(<= s1 1)", truth="fails",
      note="at most one reader -- true of the *writer*, and the natural "
           "mistake to make about the reader. Refuted by two firings of "
           "`start_read`: there are three permits and a reader takes one"),

    # ───────────────────── dining philosophers ────────────────────
    C(net="philo", name="one-eats", kind="safety",
      P="(<= (+ s0 s1 s2) 1)", truth="holds",
      note="at most one philosopher eats. With three seats every pair is "
           "adjacent, so the bound is global -- but it follows from the three "
           "fork semiflows together, not from any one of them, which makes it "
           "the suite's widest conjunctive invariant"),
    C(net="philo", name="forks", kind="safety",
      P="(= (+ s3 s4 s5 (* 2 (+ s0 s1 s2))) 3)", truth="holds",
      note="the three forks, counted wherever they are: free, or held by an "
           "eater who took two at once. The weight 2 is the `eat` "
           "transitions' arc count"),
    C(net="philo", name="never-eats", kind="safety",
      P="(= s0 0)", truth="fails",
      note="philosopher 0 never eats. Refuted -- `eat0` is enabled in the "
           "initial marking -- and the pair to `philo-rr/philo1-eats`, which "
           "says the opposite thing about a different philosopher and holds"),
    C(net="philo-rr", name="philo0-eats", kind="buchi",
      P="(= s0 1)", truth="holds",
      note="under round-robin, philosopher 0 eats infinitely often -- it "
           "reaches the forks first every cycle"),
    C(net="philo-rr", name="philo1-eats", kind="buchi",
      P="(= s1 1)", truth="fails",
      note="philosopher 1 eats infinitely often. Refuted: round-robin starves "
           "it, because philosopher 0 has already taken a shared fork by the "
           "time its own transition is tried. A fair scheduler is not a "
           "*fair* scheduler, and this is the row that says so"),

    # ─────────────────── inhibitor arcs ───────────────────────────
    C(net="inhibit", name="cpu", kind="safety",
      P="(<= (+ s2 s3) 1)", truth="holds",
      note="at most one job runs. An ordinary semaphore argument, here only "
           "to show that the inhibitor arcs have not broken it"),
    C(net="inhibit", name="queue-bound", kind="safety",
      P="(and (<= s0 2) (<= s1 2))", truth="holds",
      note="both queues stay within two. This holds **because of the "
           "inhibitor arcs and for no other reason**: each arrival is a "
           "source transition, which in a classical net is always enabled and "
           "makes its target place unbounded. Drop the inhibitor and the "
           "property is not merely unproved, it is false"),
    C(net="inhibit", name="queue-tight", kind="safety",
      P="(<= s0 1)", truth="fails",
      note="the high-priority queue bound, off by one. Refuted at hq = 2, "
           "which is exactly the value the inhibitor arc permits -- so a "
           "route has to get the inhibitor's threshold right rather than "
           "merely notice that the queue is bounded"),

    # ──────────────────────── reset arcs ──────────────────────────
    C(net="reset", name="slots-le", kind="safety",
      P="(<= (+ s0 s1) 4)", truth="holds",
      note="`q + free <= 4`. The complementary-place semiflow, weakened to an "
           "inequality, which is all a reset net leaves of it"),
    C(net="reset", name="slots-eq", kind="safety",
      P="(= (+ s0 s1) 4)", truth="fails",
      note="the same semiflow as an equality -- exactly what `prodcons/slots` "
           "asserts, in a net with one extra arc. Refuted at (q, free, done) "
           "= (0, 3, 4): `abort` discarded a token that the equality counts. "
           "The clearest statement in the suite of what a reset arc costs, "
           "and the row a route that reasons by token conservation gets wrong"),

    # ─────────────────────── transfer arcs ────────────────────────
    C(net="transfer", name="conserved", kind="safety",
      P="(= (+ s0 s1 s2) 4)", truth="holds",
      note="`p + q + done = 4` in a net with a transfer arc. A transfer moves "
           "every token at once but destroys none, so the semiflow that "
           "`reset/slots-eq` loses is still an equality here -- the two rows "
           "are the same claim about the two extensions"),
    C(net="transfer", name="pending", kind="safety",
      P="(= (+ s0 s1) 4)", truth="fails",
      note="conservation stated over the two working places only. Refuted "
           "once `finish` has fired: the tokens are still there, in `done`. "
           "Being conservative is not being *invariant on a subset*"),
    C(net="transfer-rr", name="drains", kind="buchi",
      P="(= s0 0)", truth="holds",
      note="under round-robin, `p` is empty infinitely often. The transfer "
           "empties it in a single step from any marking, so the ranking "
           "function does not need to count tokens -- it needs to count "
           "scheduler positions"),
    C(net="transfer-rr", name="batched", kind="buchi",
      P="(= s1 4)", truth="fails",
      note="all four tokens gathered in `q` infinitely often. Refuted: "
           "`finish` fires once per cycle and moves one token out of "
           "circulation, so after the first cycle the transfer never has four "
           "to move again. Holds for a while and then never -- the shape a "
           "bounded search is least likely to refute"),

    # ─────────────────── continuous Petri net ─────────────────────
    C(net="sem", name="integral", kind="safety",
      P="(or (= s0 0) (>= s0 1))", truth="holds",
      note="a critical section holds either no token or a whole one. Over the "
           "integers this needs only that a marking is non-negative, which is "
           "why it is here: it is the discrete half of the pair below, and the "
           "cheapest row in the suite to get right"),
    C(net="sem-cont", name="mutex", kind="safety",
      P="(<= (+ s0 s1) 1.0)", truth="holds",
      note="mutual exclusion over the *reals*. Every P-semiflow of a net is "
           "still an invariant of its continuous relaxation, so the bound "
           "survives -- over a state where each place holds a fraction"),
    C(net="sem-cont", name="semiflow", kind="safety",
      P="(= (+ s0 s1 s2) 1.0)", truth="holds",
      note="the semaphore semiflow, unchanged by the relaxation. The same "
           "equality and the same proof as `mutex/semiflow`, over an LRA "
           "module instead of an LIA one -- on this pair that sort is the "
           "only difference the matrix is measuring"),
    C(net="sem-cont", name="integral", kind="safety",
      P="(or (= s0 0.0) (>= s0 1.0))", truth="fails",
      note="`sem/integral` again, over the reals: the property that "
           "**separates a net from its continuous relaxation**. Refuted after "
           "two firings, by a marking holding a *fraction* of a token in the "
           "critical section -- 5/8 under the checked-in seed -- which no "
           "discrete firing sequence reaches. A route that verifies this has "
           "proved the relaxation exact, which it is not; and the two rows "
           "differ in nothing but the sort of a marking, so a route that "
           "answers both correctly is reading the module"),

    # ────────────────────── time Petri net ────────────────────────
    C(mod="p_timed", name="deadline", kind="safety",
      P="(and (>= s0 0.0) (<= s0 5.0))", pre="(and (> e0 0.0) (<= e0 1.0))",
      truth="holds",
      note="the clock never passes the transition's latest firing time. Holds "
           "for every time-step profile the precondition admits, because "
           "strong semantics truncates the step at the deadline rather than "
           "detecting a violation afterwards -- the invariant has to see that "
           "truncation, not just the guard"),
    C(mod="p_timed", name="completes", kind="buchi",
      P="(not s1)", pre="(and (> e0 0.0) (<= e0 1.0))", truth="holds",
      note="the job always eventually finishes. This is the row that "
           "distinguishes the *semantics*: the scheduler may decline to fire "
           "for ever, so the recurrence holds only because the transition is "
           "urgent at 5. Under weak semantics the same net refutes it"),
    C(mod="p_timed", name="early", kind="safety",
      P="(<= s0 4.0)", pre="(and (> e0 0.0) (<= e0 1.0))", truth="fails",
      note="a clock bound inside the firing window. Refuted at c = 4.125 -- a "
           "value reached only by a *fractional* time step, so the "
           "counterexample is not on the integer grid the module's constants "
           "live on"),

    # ───────────────────── hybrid Petri net ───────────────────────
    C(mod="p_fluid", name="band", kind="safety",
      P="(and (>= s0 0.0) (<= s0 9.0))", pre="(and (>= e0 0.875) (<= e0 1.0))",
      truth="holds",
      note="the buffer neither underflows nor overflows, for every drain rate "
           "in [7/8, 1]. The lower bound holds by the clamp -- a continuous "
           "place moves what it has -- and the upper one only after two ticks "
           "of lag are accounted for: the run peaks at 33/4, above both "
           "thresholds in the module"),
    C(mod="p_fluid", name="empties", kind="buchi",
      P="(<= s0 0.0)", pre="(and (>= e0 0.875) (<= e0 1.0))", truth="holds",
      note="the buffer is empty infinitely often, and *exactly* empty: the "
           "clamp lands the level on 0 rather than approaching it. A "
           "recurrence that a route can only prove by reasoning about the "
           "`min`, and one whose ranking function must work for the whole "
           "rate interval"),
    C(mod="p_fluid", name="tight", kind="safety",
      P="(<= s0 8.0)", pre="(and (>= e0 0.875) (<= e0 1.0))", truth="fails",
      note="a ceiling above both of the module's own thresholds, and still "
           "wrong. Refuted at 33/4 = 8.25: the feeder runs a tick past 7 and "
           "the drain starts a tick after 15/2, and the two lags compound"),
]
