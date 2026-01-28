from torch import IntTensor

from zrth import DType
from zrth.expr import nxt, ite, Sym
from zrth import ReactiveModule
import zrth.bmc as bmc


class MyModule(ReactiveModule):
    def init(self, extl):
        # extl is a vector with dimension 2
        return IntTensor([[0, 0], [1, 0], [0, 1]]) @ nxt(extl)

    def update(self, state, inp):
        # state = (x, y, z) is a vector with dimension 3,
        # inp = (y0, z0) is a vector with dimension 2
        result1 = state + IntTensor([1, 0, 0])
        result2 = IntTensor([[0, 0, 0], [0, 1, 0], [0, 0, 1]]) @ state
        x = IntTensor([1, 0, 0]) @ state
        y = IntTensor([0, 1, 0]) @ state
        z = IntTensor([0, 0, 1]) @ state

        cond = (x < y) or (x < z)
        return ite(cond, result1, result2)


def test_unroll_manual():
    """
    Manual module unrolling
    """

    m = MyModule(intf="xyz: Tensor<3; Int>", extl="yz0: Tensor<2; Int>")
    assert m

    # get the module terms
    init_terms = [term for atom in m.atoms for term in atom.init()]
    update_terms = [term for atom in m.atoms for term in atom.update()]

    transitions = [init_terms]
    # track used symbols for printing (but eventually for later usage)
    interfaces = [[(s, s.nxt()) for s in m.extl]]

    N = 4

    # chain the `update` N times
    last_env: list[Sym] = [s.nxt() for s in m.extl]
    last_out: list[Sym] = [s.nxt() for s in m.ctrl]
    for i in range(1, N + 1):
        renaming = {}

        # create new environment variables
        new_env = [Sym(f"{s.nxt().name}_{i}", s.dtype(), assoc=False) for s in m.extl]

        # create new output wires
        new_out: list[Sym] = [
            Sym(f"{s.nxt().name}_{i}", s.dtype(), assoc=False) for s in m.ctrl
        ]

        # rewire the environment wires
        renaming.update({e.wire(): l for e, l in zip(m.extl, last_env)})
        renaming.update({e.nxt().wire(): n for e, n in zip(m.extl, new_env)})

        # wire inputs of `update` to outputs of the last `transitions`
        renaming.update({s.wire(): lw.wire() for s, lw in zip(m.ctrl, last_out)})

        # rename the new outputs
        renaming.update({s.nxt().wire(): o.wire() for s, o in zip(m.ctrl, new_out)})

        new_terms = []
        for term in update_terms:
            new_term, renaming = term.remap(renaming)
            new_terms.append(new_term)

        transitions.append(new_terms)

        interfaces.append([(o, n) for o, n in zip(last_out, new_out)])
        last_out = new_out

    # ##########
    # print it!
    # ##########
    for n in range(len(transitions)):
        print("-----")
        for l, u in interfaces[n]:
            print(f"{l} = w{l.wire().id()}")
            print(f"{u} = w{u.wire().id()}")
        print("-----")
        for term in transitions[n]:
            print(term)

    # U.dbg()


def test_unroll_semimanual():
    """
    Semi-manual module unrolling
    """

    m = MyModule(intf="xyz: Tensor<3; Int>", extl="yz0: Tensor<2; Int>")
    assert m

    U = bmc.WiredTransitions()
    T = m.init_as_transition()

    # wire in the initial transition (since the unrolling is empty,
    # this is basically `push`)
    U.wire_transition(T)

    T = m.update_as_transition()
    # wire the `update` transition 3x to the unrolling
    for i in range(3):
        U.wire_transition(T)

    # U.dbg()


def test_unroll_auto():
    """
    Automated module unrolling
    """

    m = MyModule(intf="xyz: Tensor<3; Int>", extl="yz0: Tensor<2; Int>")
    assert m
    # m.to_html("/tmp/torch.html", open=True)

    U = bmc.ModuleUnrolling(m)
    U.init()
    for i in range(10):
        U.step()
    # U.dbg()
    # U.to_html("/tmp/unroll.html", open=True)


if __name__ == "__main__":
    test_unroll_auto()
    test_unroll_semimanual()
    test_unroll_manual()
