from .environments import SimpleEnv, GridWorldEnv, ComplexDecisionEnv, EarlyReturnEnv, ComparisonChainEnv, TwoBitCounterEnv
from .qnetworks import SimpleQNet, GridWorldQNet
from zrth import Module


def simpleqnet():
    qnet = SimpleQNet(
        state_size=1,
        action_size=2,
        hidden_size=2,
    )

    print("QNetwork reactive module:")
    print(qnet)

    return qnet


def test_simpleqnet_conversion():
    _ = simpleqnet()


def simpleenv():
    env = SimpleEnv()

    print("\nSimpleEnv reactive module:")
    print(env)

    return env


def test_simpleenv_conversion():
    _ = simpleenv()


def gridworldqnet():
    qnet = GridWorldQNet(
        state_size=1,
        action_size=4,
        hidden_size1=8,
        hidden_size2=4,
    )

    print("GridWorldQNet reactive module:")
    print(qnet)

    return qnet


def test_gridworldqnet_conversion():
    _ = gridworldqnet()


def gridworldenv():
    env = GridWorldEnv()

    print("\nGridWorldEnv reactive module:")
    print(env)

    return env


def test_gridworldenv_conversion():
    _ = gridworldenv()


def complexdecisionenv():
    env = ComplexDecisionEnv()

    print("\nComplexDecisionEnv reactive module:")
    print(env)

    return env


def test_complexdecisionenv_conversion():
    _ = complexdecisionenv()


def earlyreturnenv():
    env = EarlyReturnEnv()

    print("\nEarlyReturnEnv reactive module:")
    print(env)

    return env


def test_earlyreturnenv_conversion():
    _ = earlyreturnenv()


def comparisonchainenv():
    env = ComparisonChainEnv()

    print("\nComparisonChainEnv reactive module:")
    print(env)

    return env


def test_comparisonchainenv_conversion():
    _ = comparisonchainenv()


def twobitcounterenv():
    env = TwoBitCounterEnv()

    print("\nTwoBitCounterEnv reactive module:")
    print(env)

    return env


def test_twobitcounterenv_conversion():
    _ = twobitcounterenv()


if __name__ == "__main__":
    qnet = simpleqnet()
    print("\n" + "="*60 + "\n")
    env = simpleenv()
    print("\n" + "="*60 + "\n")
    composed = Module.parallel(qnet, env)
    print(composed)
    
    print("\n" + "="*60 + "\n")
    qnet3 = gridworldqnet()
    print("\n" + "="*60 + "\n")
    gridenv = gridworldenv()
    print("\n" + "="*60 + "\n")
    composed2 = Module.parallel(qnet3, gridenv)
    print(composed2)
    
    print("\n" + "="*60 + "\n")
    complexenv = complexdecisionenv()
    
    print("\n" + "="*60 + "\n")
    earlyenv = earlyreturnenv()
    
    print("\n" + "="*60 + "\n")
    compenv = comparisonchainenv()

    print("\n" + "="*60 + "\n")
    bitco = twobitcounterenv()

