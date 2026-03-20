from .environments import (
    SimpleEnv,
    GridWorldEnv,
    ComplexDecisionEnv,
    EarlyReturnEnv,
    ComparisonChainEnv,
    TwoBitCounterEnv,
    HeartODE,
    ArrayEnv,
)
from .qnetworks import SimpleQNet, GridWorldQNet
from zrth import Module, Env, NN


def simpleqnet():
    nn_module = SimpleQNet(
        state_size=1,
        action_size=2,
        hidden_size=2,
    )
    wrapped = NN(nn_module)

    print("QNetwork reactive module:")
    print(wrapped)

    return wrapped


def test_simpleqnet_conversion():
    _ = simpleqnet()


def simpleenv():
    env = SimpleEnv()
    wrapped = Env(env)

    print("\nSimpleEnv reactive module:")
    print(wrapped)

    return wrapped


def test_simpleenv_conversion():
    _ = simpleenv()


def gridworldqnet():
    nn_module = GridWorldQNet(
        state_size=1,
        action_size=4,
        hidden_size1=8,
        hidden_size2=4,
    )
    wrapped = NN(nn_module)

    print("GridWorldQNet reactive module:")
    print(wrapped)

    return wrapped


def test_gridworldqnet_conversion():
    _ = gridworldqnet()


def gridworldenv():
    env = GridWorldEnv()
    wrapped = Env(env)

    print("\nGridWorldEnv reactive module:")
    print(wrapped)

    return wrapped


def test_gridworldenv_conversion():
    _ = gridworldenv()


def complexdecisionenv():
    env = ComplexDecisionEnv()
    wrapped = Env(env)

    print("\nComplexDecisionEnv reactive module:")
    print(wrapped)

    return wrapped


def test_complexdecisionenv_conversion():
    _ = complexdecisionenv()


def earlyreturnenv():
    env = EarlyReturnEnv()
    wrapped = Env(env)

    print("\nEarlyReturnEnv reactive module:")
    print(wrapped)

    return wrapped


def test_earlyreturnenv_conversion():
    _ = earlyreturnenv()


def comparisonchainenv():
    env = ComparisonChainEnv()
    wrapped = Env(env)

    print("\nComparisonChainEnv reactive module:")
    print(wrapped)

    return wrapped


def test_comparisonchainenv_conversion():
    _ = comparisonchainenv()


def twobitcounterenv():
    env = TwoBitCounterEnv()
    wrapped = Env(env)

    print("\nTwoBitCounterEnv reactive module:")
    print(wrapped)

    return wrapped


def test_twobitcounterenv_conversion():
    _ = twobitcounterenv()


def heartode():
    env = HeartODE(60.0, 5.0, 1.0, 0.1)
    wrapped = Env(env)

    print("\nHeartODE reactive module:")
    print(wrapped)

    return wrapped


def test_heartode_conversion():
    _ = heartode()


def test_arrayenv_conversion():
    env = ArrayEnv()
    wrapped = Env(env)
    module_str = str(wrapped)
    if "Float(3, 3)" not in module_str:
        raise AssertionError(module_str)  # grid
    if "Float(5)" not in module_str:  # weights
        raise AssertionError(module_str)  # grid
    if "Float(2, 2)" not in module_str:  # matrix
        raise AssertionError(module_str)  # grid


if __name__ == "__main__":
    qnet = simpleqnet()
    print("\n" + "=" * 60 + "\n")
    env = simpleenv()
    print("\n" + "=" * 60 + "\n")
    composed = Module.parallel(qnet, env)
    print(composed)

    print("\n" + "=" * 60 + "\n")
    qnet3 = gridworldqnet()
    print("\n" + "=" * 60 + "\n")
    gridenv = gridworldenv()
    print("\n" + "=" * 60 + "\n")
    composed2 = Module.parallel(qnet3, gridenv)
    print(composed2)

    print("\n" + "=" * 60 + "\n")
    complexenv = complexdecisionenv()

    print("\n" + "=" * 60 + "\n")
    earlyenv = earlyreturnenv()

    print("\n" + "=" * 60 + "\n")
    compenv = comparisonchainenv()

    print("\n" + "=" * 60 + "\n")
    bitco = twobitcounterenv()

    print("\n" + "=" * 60 + "\n")
    heart = heartode()
