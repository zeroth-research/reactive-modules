"""Every property the `hybrid` suite asks, and what is known about it.

One row is one question: a module in this directory, a property over its
state components `s0..sN-1`, and -- for the modules with inputs -- the
precondition that makes the question well posed.

`truth` is what `Row.truth` carries into the matrix, and it decides whether a
route's answer is *correct*: only a row whose property fails may honestly be
`REFUTED`. Every value here is checked by `tests/bench_matrix/test_rows.py`,
which steps the module and looks at the states it reaches -- so a `fails` row
names a state that really is reachable, and a `holds` row has at least not
been refuted. The bounds quoted in the notes are from that simulation, over
runs far longer than the test's own.

Each module contributes at least one row that **fails**. A suite of only-true
properties measures half a route: it can never show that a refutation is
right, and a route that answers `VERIFIED` to everything would score
perfectly on it.
"""

C = dict

CASES = [
    # ───────────────────────── thermostat ─────────────────────────
    C(mod="m_thermostat", name="band", kind="safety",
      P="(and (>= s0 15.0) (<= s0 25.0))", truth="holds",
      note="the safe temperature band. The run covers [16.70, 23.22], so the "
           "band has slack on both sides -- but the *inductive* invariant does "
           "not: a plain interval is not inductive here, because from T = 25 "
           "with the heater on the next temperature is 25.5. Anything that "
           "proves this has to relate T to the heater's mode"),
    C(mod="m_thermostat", name="warms", kind="buchi",
      P="(>= s0 21.0)", truth="holds",
      note="the heater always eventually warms the room. Holds because the "
           "hysteresis cannot stall: the run is not periodic in any bounded "
           "window (0.875 has no finite orbit on these values), so the "
           "ranking function cannot be a cycle counter"),
    C(mod="m_thermostat", name="setpoint", kind="safety",
      P="(<= s0 22.0)", truth="fails",
      note="the upper *hysteresis* setpoint, mistaken for a safety bound. "
           "Refuted at T = 22.20: the controller samples, so it overshoots by "
           "one tick of heating. A route that verifies this has proved "
           "something false"),

    # ───────────────────────── water tank ─────────────────────────
    C(mod="m_watertank", name="band", kind="safety",
      P="(and (>= s0 5.0) (<= s0 15.0))", truth="holds",
      note="the safe level band. The run covers [5.5, 14.5] -- two and a half "
           "units outside the controller's own [6, 12], which is what a tick "
           "of actuation lag costs at these flow rates"),
    C(mod="m_watertank", name="refills", kind="buchi",
      P="(>= s0 12.0)", truth="holds",
      note="the tank always eventually refills. The dual of `drains`, and the "
           "half that needs the *filling* branch of the ranking function"),
    C(mod="m_watertank", name="setpoint", kind="safety",
      P="(<= s0 12.0)", truth="fails",
      note="the controller's stop level as a safety bound. Refuted at h = 13: "
           "the same one-tick overshoot as the thermostat's `setpoint`, in a "
           "module whose state is otherwise entirely different"),

    # ──────────────────── water tank, with a disturbance ──────────
    C(mod="m_tank_dist", name="band", kind="safety",
      P="(and (>= s0 4.0) (<= s0 15.0))", pre="(and (>= e0 0.0) (<= e0 0.25))",
      truth="holds",
      note="the band, now against every consumption profile in [0, 1/4] "
           "rather than one. The run covers [4.63, 14.88]; the property is "
           "about the whole input-quantified family, which is what makes it a "
           "different question from `m_watertank/band` and not just a noisier "
           "one"),
    C(mod="m_tank_dist", name="band-nopre", kind="safety",
      P="(and (>= s0 4.0) (<= s0 15.0))", truth="fails",
      note="the same module and the same property with `--pre` dropped, so "
           "the consumption is unbounded and may be negative -- an inflow. "
           "Refuted at h = 16.88. The pair is the limit matrix's "
           "`ReluInput`/`ReluInputNoPre` lesson on a continuous plant: the "
           "precondition is part of the question, not a hint"),
    C(mod="m_tank_dist", name="refills", kind="buchi",
      P="(>= s0 12.0)", pre="(and (>= e0 0.0) (<= e0 0.25))", truth="holds",
      note="recurrence under a disturbance: the ranking function has to "
           "decrease for *every* admissible consumption, so it cannot be read "
           "off a single trajectory"),

    # ───────────────────────── two tanks ──────────────────────────
    C(mod="m_twotanks", name="box", kind="safety",
      P="(and (>= s0 1.0) (<= s0 7.0) (>= s1 0.0) (<= s1 9.0))", truth="holds",
      note="a box over both levels: [1.5, 6.5] and [0.75, 8.75]. The two "
           "loops have different periods, so the box is not a product of two "
           "one-dimensional arguments -- tank 2 leaves its own hysteresis "
           "band by two and a half units, and only tank 1's pump explains it"),
    C(mod="m_twotanks", name="tank1-full", kind="buchi",
      P="(>= s0 6.0)", truth="holds",
      note="tank 1 always eventually reaches its high mark. True of tank 1 "
           "alone, which is the point of the pair below"),
    C(mod="m_twotanks", name="both-full", kind="buchi",
      P="(and (>= s0 6.0) (>= s1 6.0))", truth="fails",
      note="both tanks full *at the same tick*. Refuted: each is high "
           "infinitely often, and `G F (p and q)` is not `G F p and G F q`. "
           "The conjunction of two true recurrences is the mistake this row "
           "exists to catch"),

    # ──────────────────────── room heating ────────────────────────
    C(mod="m_roomheat", name="box", kind="safety",
      P="(and (>= s0 14.0) (<= s0 22.0) (>= s1 12.0) (<= s1 22.0))",
      truth="holds",
      note="a box over both rooms: [16.07, 20.79] and [14.18, 19.91]. Neither "
           "room's bound survives on its own -- each room's temperature is a "
           "term in the other's update -- so this is a genuinely relational "
           "invariant over two Reals and a mode bit"),
    C(mod="m_roomheat", name="room1-warm", kind="buchi",
      P="(>= s0 18.0)", truth="holds",
      note="room 1 is always eventually warm. It is warm only while the "
           "heater is there, and the heater leaves, so the ranking function "
           "has to count down to the heater's *return*"),
    C(mod="m_roomheat", name="room2-warm", kind="buchi",
      P="(>= s1 18.0)", truth="holds",
      note="the same claim for the draughty room, which loses heat to the "
           "outside twice as fast and therefore spends longer below the "
           "threshold. Same property shape, strictly harder ranking"),
    C(mod="m_roomheat", name="both-warm", kind="buchi",
      P="(and (>= s0 18.0) (>= s1 18.0))", truth="fails",
      note="both rooms warm at once. Refuted -- one heater cannot hold two "
           "rooms above the threshold -- and refuted for a *physical* reason "
           "rather than a scheduling one, unlike `m_twotanks/both-full`"),

    # ──────────────────────── bouncing ball ───────────────────────
    C(mod="m_bounce", name="above-floor", kind="safety",
      P="(and (>= s0 0.0) (<= s0 4.0))", truth="holds",
      note="the ball stays between the floor and its release height. The "
           "lower bound holds by the reset rather than by the dynamics, and "
           "the upper one because Euler loses energy at every bounce -- a "
           "discretisation that gained energy would refute it, which is the "
           "usual failure of a naive bouncing-ball encoding"),
    C(mod="m_bounce", name="lands", kind="buchi",
      P="(<= s0 0.5)", truth="holds",
      note="the ball always eventually returns to the floor. The velocity at "
           "the floor converges to 1/3, a value no constant in the module "
           "names, so the recurrence has no eventual period to count"),
    C(mod="m_bounce", name="speed", kind="safety",
      P="(>= s1 -5.0)", truth="fails",
      note="a bound on the downward velocity. Refuted at v = -6, on the first "
           "fall: the ball is dropped from 4 and gathers six ticks of gravity "
           "before it lands. A bound that is right for every later bounce and "
           "wrong for the transient"),

    # ──────────────────── adaptive cruise control ─────────────────
    C(mod="m_cruise", name="no-collision", kind="safety",
      P="(>= s0 1.0)", truth="holds",
      note="the follower never reaches the leader. The gap's minimum is 3.70, "
           "reached on the first undershoot while the actuator is saturated; "
           "the linear part of the loop says nothing about that stretch"),
    C(mod="m_cruise", name="settles", kind="buchi",
      P="(and (>= s0 4.5) (<= s0 5.5))", truth="holds",
      note="the gap always eventually sits within half a unit of its set "
           "distance. The closed loop's eigenvalues are complex, so the gap "
           "crosses the band rather than approaching it from one side, and a "
           "monotone ranking function in the gap alone cannot show it"),
    C(mod="m_cruise", name="overshoot", kind="safety",
      P="(>= s0 4.0)", truth="fails",
      note="a gap bound that the steady state satisfies and the transient "
           "does not. Refuted on the way in, where the gap undershoots to "
           "3.70 -- the overshoot the saturated approach causes"),

    # ───────────────────────── reactor rods ───────────────────────
    C(mod="m_reactor", name="no-shutdown", kind="safety",
      P="(not s5)", truth="holds",
      note="the reactor never has to be shut down: it never passes 545 with "
           "both rods locked out. This is the benchmark's own property, and "
           "it holds by a margin of one tick -- at the overheat that rod 2 "
           "answers, rod 1's timer stands at 11 against a window of 12. No "
           "invariant that drops either timer can prove it"),
    C(mod="m_reactor", name="temp-band", kind="safety",
      P="(and (>= s0 450.0) (<= s0 570.0))", truth="holds",
      note="the core temperature stays in [451.998, 563.688] over 40k ticks. "
           "The bound is a consequence of `no-shutdown` plus two ticks of "
           "overshoot at each end, so it is the same argument with the "
           "arithmetic left in"),
    C(mod="m_reactor", name="cools", kind="buchi",
      P="(<= s0 520.0)", truth="holds",
      note="the core always eventually cools below 520. Needs a rod to become "
           "available, so the ranking function has to run over a timer, not "
           "over the temperature"),
    C(mod="m_reactor", name="temp-tight", kind="safety",
      P="(<= s0 560.0)", truth="fails",
      note="a temperature ceiling ten degrees above the rod threshold. "
           "Refuted at 561.71, and only on the overheat that rod 2 answers: "
           "the run passes 560 exactly when the *slower* rod is the one "
           "available, so a route has to reach the second cycle to find it"),
]
