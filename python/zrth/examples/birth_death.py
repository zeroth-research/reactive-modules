"""A birth-death process as a stochastic Petri net: births at rate 2, deaths at rate 1.

Each transition is a module owning a Poisson clock and a "fires now" flag; the place
is a module counting tokens as it awaits the flags. Clocks run down against the external
time reference ``t`` (``-1 * d(t)``) and are re-armed with a fresh Exponential delay when
they expire; the death clock only runs while there is a token to consume.
"""

from zrth import SPN, Event, Clock, Nat, Var
from zrth import Module as compose
from zrth.sugar import Module, X, d, ite, exp

t, bclk, dclk = Var(Clock()), Var(Clock()), Var(Clock())
bth, dth = Var(Event()), Var(Event())
n = Var(Nat())


class Birth(Module):  # a birth at 2 Hz
    def init(self, t):
        return exp(2.0), False

    def next(self, clk, bth, t):
        return ite(clk == 0, exp(2.0), clk), clk == 0

    def flow(self, clk, bth, t):
        return -1 * d(t), None


class Death(Module):  # a death at 1 Hz, while there are tokens
    def init(self, n, t):
        return exp(1.0), False

    def next(self, clk, dth, n, t):
        fires = (clk == 0) & (n != 0)
        return ite(fires, exp(1.0), clk), fires

    def flow(self, clk, dth, n, t):
        return ite(n != 0, -1 * d(t), 0 * d(t)), None


class Place(Module):  # the token count
    def init(self, bth, dth):
        return 0

    def next(self, n, bth, dth):
        return ite(X(bth) & ~X(dth), n + 1, ite(X(dth) & ~X(bth) & (n != 0), n - 1, n))


birth = Birth(theory=SPN, ctrl=(bclk, bth), extl=(t,))
death = Death(theory=SPN, ctrl=(dclk, dth), extl=(n, t))
place = Place(theory=SPN, ctrl=(n,), extl=(bth, dth))
system = compose(birth, death, place, hide={bclk, dclk})


<<<<<<< HEAD
def visual():
=======
def syst():
>>>>>>> 69b049f ([SQUEEZE] minor changes)
    return compose(birth, death, place, hide={bclk, dclk})


if __name__ == "__main__":
    print(
        system.with_varnames(
            {t: "t", bclk: "bclk", dclk: "dclk", bth: "bth", dth: "dth", n: "n"}
        )
    )
