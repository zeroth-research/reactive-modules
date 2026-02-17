from zrth.gym import SimpleQNet, SimpleEnv, GridWorldEnv, GridWorldQNet, ComplexDecisionEnv, EarlyReturnEnv, ComparisonChainEnv
from zrth import reset_ctx, Module


def simpleqnet():
    qnet = SimpleQNet(
        extl=["observation: Tensor<1; Float>"],
        intf=["q_values: Tensor<2; Float>"],
        state_size=1,
        action_size=2,
        hidden_size=2,
    )

    print("QNetwork reactive module:")
    print(qnet.unwrap())

    return qnet


def test_simpleqnet_conversion():
    _ = simpleqnet()


def simpleenv():
    env = SimpleEnv(
        extl=["q_values: Tensor<2; Float>"],
        intf=[
            "observation: Tensor<1; Float>",
            "reward: Tensor<1; Float>",
            "terminated: Tensor<1; Bool>",
            "truncated: Tensor<1; Bool>",
        ],
        prvt=["state: Tensor<1; Float>"],
    )

    print("\nSimpleEnv reactive module:")
    print(env.unwrap())

    return env


def test_simpleenv_conversion():
    _ = simpleenv()


def gridworldqnet():
    qnet = GridWorldQNet(
        extl=["observation: Tensor<1; Float>"],
        intf=["q_values: Tensor<4; Float>"],
        state_size=1,
        action_size=4,
        hidden_size1=8,
        hidden_size2=4,
    )

    print("GridWorldQNet reactive module:")
    print(qnet.unwrap())

    return qnet


def test_gridworldqnet_conversion():
    _ = gridworldqnet()


def gridworldenv():
    env = GridWorldEnv(
        extl=["q_values: Tensor<4; Float>"],
        intf=[
            "observation: Tensor<1; Float>",
            "reward: Tensor<1; Float>",
            "terminated: Tensor<1; Bool>",
            "truncated: Tensor<1; Bool>",
        ],
        prvt=[
            "x: Tensor<1; Float>",
            "y: Tensor<1; Float>",
        ],
    )

    print("\nGridWorldEnv reactive module:")
    print(env.unwrap())

    return env


def test_gridworldenv_conversion():
    _ = gridworldenv()


def complexdecisionenv():
    env = ComplexDecisionEnv(
        extl=["q_values: Tensor<10; Float>"],
        intf=[
            "observation: Tensor<1; Float>",
            "reward: Tensor<1; Float>",
            "terminated: Tensor<1; Bool>",
            "truncated: Tensor<1; Bool>",
        ],
        prvt=[
            "score: Tensor<1; Float>",
            "multiplier: Tensor<1; Float>",
            "bonus_active: Tensor<1; Bool>",
        ],
    )

    print("\nComplexDecisionEnv reactive module:")
    print(env.unwrap())

    return env


def test_complexdecisionenv_conversion():
    _ = complexdecisionenv()


def earlyreturnenv():
    env = EarlyReturnEnv(
        extl=["q_values: Tensor<5; Float>"],
        intf=[
            "observation: Tensor<1; Float>",
            "reward: Tensor<1; Float>",
            "terminated: Tensor<1; Bool>",
            "truncated: Tensor<1; Bool>",
        ],
        prvt=["counter: Tensor<1; Float>"],
    )

    print("\nEarlyReturnEnv reactive module:")
    print(env.unwrap())

    return env


def test_earlyreturnenv_conversion():
    _ = earlyreturnenv()


def comparisonchainenv():
    env = ComparisonChainEnv(
        extl=["q_values: Tensor<3; Float>"],
        intf=[
            "observation: Tensor<1; Float>",
            "reward: Tensor<1; Float>",
            "terminated: Tensor<1; Bool>",
            "truncated: Tensor<1; Bool>",
        ],
        prvt=["value: Tensor<1; Float>"],
    )

    print("\nComparisonChainEnv reactive module:")
    print(env.unwrap())

    return env


def test_comparisonchainenv_conversion():
    _ = comparisonchainenv()


if __name__ == "__main__":
    _ = reset_ctx()
    qnet = simpleqnet()
    print("\n" + "="*60 + "\n")
    env = simpleenv()
    print("\n" + "="*60 + "\n")
    composed = Module.parallel(qnet.unwrap(), env.unwrap())
    print(composed)
    
    print("\n" + "="*60 + "\n")
    print("=" * 60)
    qnet3 = gridworldqnet()
    
    print("\n" + "="*60 + "\n")
    print("=" * 60)
    gridenv = gridworldenv()
    
    print("\n" + "="*60 + "\n")
    print("=" * 60)
    composed2 = Module.parallel(qnet3.unwrap(), gridenv.unwrap())
    print(composed2)
    
    print("\n" + "="*60 + "\n")
    print("=" * 60)
    complexenv = complexdecisionenv()
    
    print("\n" + "="*60 + "\n")
    print("=" * 60)
    print("Testing EarlyReturnEnv (early returns in if/else)")
    earlyenv = earlyreturnenv()
    
    print("\n" + "="*60 + "\n")
    print("=" * 60)
    print("Testing ComparisonChainEnv (comparison chains like 0 < x < 10)")
    compenv = comparisonchainenv()
