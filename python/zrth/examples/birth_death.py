"""A birth-death process as a stochastic Petri net: births at rate 2, deaths at rate 1.

Each transition is a module owning a clock and a firing event; the place is a module
counting tokens as it awaits the events. A clock runs down against the external time
reference ``t`` (``-1 * d(t)``) for as long as it is non-negative, and past that has no
flow at all (``ite(clk >= 0, ..., None)``: a state one cannot be in, so time stops at the
expiry and that is where the round happens); it is re-armed with a fresh Exponential
delay when it expires. An event does not move (its flow is ``0``) and fires by changing
value: a transition toggles it (``~e``) on the steps where it fires, and the place reads
the toggle with ``fired``. The place alone decides what a firing does: a death on an
empty place is ignored.
"""

from zrth import SPN, Event, Clock, Nat, Var
from zrth import Module as compose
from zrth.sugar import Module, d, ite, fired, exp

t, bclk, dclk = Var(Clock()), Var(Clock()), Var(Clock())
bth, dth = Var(Event()), Var(Event())
n = Var(Nat())


class Birth(Module):
    def init(self, t):
        return exp(2.0), False

    def next(self, clk, bth, t):
        fires = clk == 0
        return ite(fires, exp(2.0), clk), ite(fires, ~bth, bth)

    def flow(self, clk, bth, t):
        return ite(clk >= 0, -1 * d(t), None), 0


class Death(Module):
    def init(self, n, t):
        return exp(1.0), False

    def next(self, clk, dth, n, t):
        fires = (clk == 0)
        return ite(fires, exp(1.0), clk), ite(fires, ~dth, dth)

    def flow(self, clk, dth, n, t):
        return ite(clk >= 0, -1 * d(t), None), 0


class Place(Module):
    def init(self, bth, dth):
        return 0

    def next(self, n, bth, dth):
        born, died = fired(bth), fired(dth)
        return ite(born & ~died, n + 1, ite(died & ~born & (n != 0), n - 1, n))


birth = Birth(theory=SPN, ctrl=(bclk, bth), extl=(t,))
death = Death(theory=SPN, ctrl=(dclk, dth), extl=(n, t))
place = Place(theory=SPN, ctrl=(n,), extl=(bth, dth))
system = compose(birth, death, place, hide={bclk, dclk})


def visual():
    return compose(birth, death, place, hide={bclk, dclk})


if __name__ == "__main__":
    print(
        system.with_varnames(
            {t: "t", bclk: "bclk", dclk: "dclk", bth: "bth", dth: "dth", n: "n"}
        )
    )
