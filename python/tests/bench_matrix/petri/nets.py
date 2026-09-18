"""Every Petri net the `petri` suite measures, as data.

A place/transition net is a marking plus a flow relation and nothing else, so
the nets here are a table rather than a file each: `petri_mod.py` turns any
row of it into the reactive module `verith` loads. What varies between rows is
only the arcs -- including the arcs that put a net *outside* the classical
class, which is the point of half of them.

## The encoding

A marking is one 1x1 `Int` variable per place (`Real`, for the one continuous
net), in `Net.places` order, so the property over `s0..sN-1` reads off the
place list directly. Firing is a step of the module:

  * **which** transition fires is an *external input*. A reactive module
    resamples its inputs every tick and `--pre` is all that constrains them,
    so a property holding of the module is a property holding for **every**
    firing sequence -- which is the verification question, not a simulation
    of one schedule. `Net.sched == "free"` is this.
  * a selection whose transition is **dead** in the current marking leaves
    the marking alone. An idle step adds no reachable marking (the marking
    is already there), so the reachability set is the net's own; what it
    buys is a total transition relation, which a `--safety` obligation over
    `X(state)` needs.
  * `Net.sched == "rr"` replaces the input with a **round-robin counter**, an
    extra last state component. That closes the module, and a closed module
    is what makes `G F p` a question about the net rather than about the
    scheduler: under a free input `G F p` is false for almost any `p`,
    because the input may select a dead transition forever.

## The extensions

Each is one field on `Trans`, and each is a strict increase in power over the
classical net -- with a property in `cases.py` that holds *because* of it, or
a classical invariant that it breaks:

  * `inh`   inhibitor arcs: the transition fires only while `m[p] < w`.
            Zero-testing makes the net Turing-powerful (Agerwala), and here
            it bounds a place that no ordinary arc can bound.
  * `reset` reset arcs: these places drop to 0. Reset nets keep coverability
            decidable but destroy tokens, so a P-semiflow survives only as an
            inequality (Dufourd-Finkel-Schnoebelen).
  * `xfer`  transfer arcs: `{src: dst}` moves *all* of `src` to `dst`.
            Transfer nets are as expressive as reset nets but conserve
            tokens, so the same semiflow that `reset` breaks still holds --
            the pair is here to be compared.
  * `real`  a continuous net (Blondin-Finkel-Haase-Haddad): markings range
            over the non-negative reals and a transition fires a *fractional*
            amount, given by a second input. Every P-semiflow of the discrete
            net still holds; the reachable set is strictly larger, and
            `cases.py` has the property that separates them.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Trans:
    """One transition: its arcs, ordinary first."""

    name: str
    pre: dict = field(default_factory=dict)     # consumed: place -> weight
    post: dict = field(default_factory=dict)    # produced: place -> weight
    inh: dict = field(default_factory=dict)     # inhibitor: fires while m[p] < w
    reset: tuple = ()                           # reset arcs: these drop to 0
    xfer: dict = field(default_factory=dict)    # transfer arcs: src -> dst


@dataclass(frozen=True)
class Net:
    places: tuple
    init: tuple
    trans: tuple
    doc: str
    source: str = ""
    sched: str = "free"        # "free": an input selects; "rr": round-robin
    real: bool = False         # a continuous net: markings over the reals

    @property
    def marking(self) -> dict:
        return dict(zip(self.places, self.init))


def _T(name, pre=None, post=None, **kw) -> Trans:
    return Trans(name, pre or {}, post or {}, **kw)


# ══════════════════════════════════════════════════════════════════════════
# Classical place/transition nets
# ══════════════════════════════════════════════════════════════════════════

MUTEX = dict(
    places=("i1", "w1", "c1", "i2", "w2", "c2", "sem"),
    init=(1, 0, 0, 1, 0, 0, 1),
    trans=(
        _T("req1", {"i1": 1}, {"w1": 1}),
        _T("enter1", {"w1": 1, "sem": 1}, {"c1": 1}),
        _T("exit1", {"c1": 1}, {"i1": 1, "sem": 1}),
        _T("req2", {"i2": 1}, {"w2": 1}),
        _T("enter2", {"w2": 1, "sem": 1}, {"c2": 1}),
        _T("exit2", {"c2": 1}, {"i2": 1, "sem": 1}),
    ),
    doc="""Two processes and a Dijkstra semaphore -- the net every
    introduction to Petri nets opens with. Each process cycles idle ->
    waiting -> critical, and entering takes the single token in `sem`. Its
    three P-semiflows (`i+w+c = 1` per process and `c1+c2+sem = 1`) are what
    mutual exclusion follows from, and they are the invariant an inferring
    route has to find.""",
    source="Peterson, `Petri Net Theory and the Modeling of Systems` (1981)",
)

PRODCONS = dict(
    places=("p_idle", "p_full", "buf", "free", "c_idle", "c_full"),
    init=(1, 0, 0, 3, 1, 0),
    trans=(
        _T("produce", {"p_idle": 1}, {"p_full": 1}),
        _T("deposit", {"p_full": 1, "free": 1}, {"p_idle": 1, "buf": 1}),
        _T("fetch", {"c_idle": 1, "buf": 1}, {"c_full": 1, "free": 1}),
        _T("consume", {"c_full": 1}, {"c_idle": 1}),
    ),
    doc="""A producer and a consumer over a buffer of three slots. `free` is
    the complementary place that bounds the buffer: a classical net cannot
    test a place for an upper bound, so the bound has to be *counted* by a
    second place, and `buf + free = 3` is the invariant that says so.""",
    source="Peterson, `Petri Net Theory and the Modeling of Systems` (1981)",
)

READWRITE = dict(
    places=("idle", "reading", "writing", "sem"),
    init=(3, 0, 0, 3),
    trans=(
        _T("start_read", {"idle": 1, "sem": 1}, {"reading": 1}),
        _T("end_read", {"reading": 1}, {"idle": 1, "sem": 1}),
        _T("start_write", {"idle": 1, "sem": 3}, {"writing": 1}),
        _T("end_write", {"writing": 1}, {"idle": 1, "sem": 3}),
    ),
    doc="""Readers and writers, three clients, as one folded net: a reader
    takes one of the three permits in `sem` and a writer takes all three.
    The weighted semiflow `reading + 3 writing + sem = 3` is the whole
    protocol -- an arc weight, not a control structure, is what excludes a
    reader and a writer at once.""",
    source="Courtois-Heymans-Parnas (1971), as the standard folded net",
)

PHILO = dict(
    places=("e0", "e1", "e2", "f0", "f1", "f2"),
    init=(0, 0, 0, 1, 1, 1),
    trans=(
        _T("eat0", {"f0": 1, "f1": 1}, {"e0": 1}),
        _T("eat1", {"f1": 1, "f2": 1}, {"e1": 1}),
        _T("eat2", {"f2": 1, "f0": 1}, {"e2": 1}),
        _T("done0", {"e0": 1}, {"f0": 1, "f1": 1}),
        _T("done1", {"e1": 1}, {"f1": 1, "f2": 1}),
        _T("done2", {"e2": 1}, {"f2": 1, "f0": 1}),
    ),
    doc="""Three dining philosophers. Each needs both of its adjacent forks,
    and with three seats every pair is adjacent, so at most one eats at a
    time -- a bound that follows from the fork semiflows rather than from any
    single place. `eat_i` takes two tokens at once, which is what makes the
    net deadlockable: the classical illustration that a safety invariant says
    nothing about progress.

    The usual `thinking` place per philosopher is folded away. It would gate
    `eat_i` on the philosopher being idle, which the forks already do -- an
    eater is holding both of them -- so the reachable markings of `e` and `f`
    are the same net either way, at two thirds of the places and, because
    `System/Circ.lean` costs the square of the block's path count, well under
    half the proof.""",
    source="Dijkstra (1971); the standard three-seat net, forks-only",
)

SEMAPHORE = dict(
    places=("c1", "c2", "sem"),
    init=(0, 0, 1),
    trans=(
        _T("enter1", {"sem": 1}, {"c1": 1}),
        _T("exit1", {"c1": 1}, {"sem": 1}),
        _T("enter2", {"sem": 1}, {"c2": 1}),
        _T("exit2", {"c2": 1}, {"sem": 1}),
    ),
    doc="""`MUTEX` with the request states folded away: a process is in its
    critical section or it is not, and entering takes the one token in `sem`.
    It keeps the same P-semiflow, `c1 + c2 + sem = 1`, and the same property.

    Three places is what it is *for*. The continuous relaxation reads its
    firing amount from a second input, which every guard and every delta then
    mentions, and `System/Circ.lean` costs the square of its path count -- so
    the seven-place net does not compile over the reals and this one does.
    `sem` and `sem-cont` are therefore a controlled pair: the only difference
    between them is the sort of a marking.""",
    source="the standard semaphore net; `MUTEX` without the waiting places",
)

# ══════════════════════════════════════════════════════════════════════════
# Extensions
# ══════════════════════════════════════════════════════════════════════════

INHIBIT = dict(
    places=("hq", "lq", "hr", "lr", "cpu"),
    init=(0, 0, 0, 0, 1),
    trans=(
        _T("h_arrive", {}, {"hq": 1}, inh={"hq": 2}),
        _T("l_arrive", {}, {"lq": 1}, inh={"lq": 2}),
        _T("h_start", {"hq": 1, "cpu": 1}, {"hr": 1}),
        _T("l_start", {"lq": 1, "cpu": 1}, {"lr": 1}, inh={"hq": 1}),
        _T("h_end", {"hr": 1}, {"cpu": 1}),
        _T("l_end", {"lr": 1}, {"cpu": 1}),
    ),
    doc="""A fixed-priority scheduler, with inhibitor arcs doing two jobs a
    classical net cannot do at all. `l_start` is inhibited by `hq`, so a
    low-priority job starts only while no high-priority job waits -- strict
    priority. And each arrival is a *source* transition, inhibited by its own
    queue: `h_arrive` fires only while `hq < 2`. In a classical net a source
    transition is always enabled and its target place is unbounded, so the
    queue bound in `cases.py` holds precisely because of the inhibitor arc
    and for no other reason.""",
    source="Agerwala-Flynn (1973), on the power of inhibitor arcs",
)

RESET = dict(
    places=("q", "free", "done"),
    init=(0, 4, 0),
    trans=(
        _T("arrive", {"free": 1}, {"q": 1}),
        _T("serve", {"q": 1}, {"free": 1, "done": 1}),
        _T("abort", reset=("q",)),
    ),
    doc="""A four-slot job queue that a supervisor can flush: `abort` is a
    reset arc, and it drops every token in `q` at once. `arrive`/`serve`
    alone give the usual complementary-place invariant `q + free = 4`; the
    reset arc *destroys* the tokens it clears, so that equality becomes an
    inequality and the net is no longer conservative. `cases.py` states both
    and expects the equality to be refuted -- the cleanest statement of what
    a reset arc costs. `done` is unbounded, so the net is not a bounded one.""",
    source="Dufourd-Finkel-Schnoebelen, ICALP'98 (reset nets)",
)

TRANSFER = dict(
    places=("p", "q", "done"),
    init=(4, 0, 0),
    trans=(
        _T("batch", xfer={"p": "q"}),
        _T("back", {"q": 1}, {"p": 1}),
        _T("finish", {"q": 1}, {"done": 1}),
    ),
    doc="""The same shape as `RESET`, with a transfer arc in place of the
    reset arc: `batch` moves *all* of `p` into `q` in one step rather than
    discarding it. Transfer nets and reset nets have the same decidability
    frontier, but a transfer arc is conservative, so `p + q + done = 4`
    survives where `RESET`'s `q + free = 4` does not. That contrast is why
    both nets are here.""",
    source="Dufourd-Finkel-Schnoebelen, ICALP'98 (transfer nets)",
)

CONTINUOUS = dict(
    doc="""`SEMAPHORE` over the non-negative reals: a continuous Petri net,
    where a transition fires a fractional amount and a place therefore holds
    a fractional number of tokens. A second input carries the amount, which
    `--pre` keeps in (0, 1], and a transition is enabled only for an amount
    its input places can actually pay for.

    Every P-semiflow of the discrete net is still an invariant -- the
    relaxation is why `cases.py` can state mutual exclusion here at all -- but
    the reachable set is strictly larger, and it is larger in a way a single
    property can pin down: the discrete net puts 0 or 1 tokens in a critical
    section, and the continuous one reaches every value between. That row is
    expected to be *refuted*, and a route that verifies it has proved
    something false.""",
    source="Blondin-Finkel-Haase-Haddad, TACAS'16 / TOCL 2017",
    real=True,
)


# `sched`/`real` variants share one net: only the scheduler or the sort of a
# marking differs, and a row's `bench` name is what tells them apart.
NETS = {
    "mutex": Net(**MUTEX),
    "mutex-rr": Net(**MUTEX, sched="rr"),
    "sem": Net(**SEMAPHORE),
    "sem-cont": Net(**{k: v for k, v in SEMAPHORE.items()
                       if k not in ("doc", "source")}, **CONTINUOUS),
    "prodcons": Net(**PRODCONS),
    "prodcons-rr": Net(**PRODCONS, sched="rr"),
    "readwrite": Net(**READWRITE),
    "philo": Net(**PHILO),
    "philo-rr": Net(**PHILO, sched="rr"),
    "inhibit": Net(**INHIBIT),
    "reset": Net(**RESET),
    "transfer": Net(**TRANSFER),
    "transfer-rr": Net(**TRANSFER, sched="rr"),
}
