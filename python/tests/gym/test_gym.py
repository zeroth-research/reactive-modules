from zrth.gym import QNetwork, SimpleEnv
from zrth import reset_ctx, Module


def qnetwork():
    qnet = QNetwork(
        extl=["observation: Tensor<1; Float>"],
        intf=["q_values: Tensor<2; Float>"],
        state_size=1,
        action_size=2,
        hidden_size=2,
    )

    print("QNetwork reactive module:")
    print(qnet._module)

    return qnet


def test_qnetwork_conversion():
    return qnetwork()


def simpleenv():
    env = SimpleEnv(
        extl=["q_values: Tensor<2; Float>"],
        intf=[
            "observation: Tensor<1; Float>",
            "reward: Tensor<1; Float>",
            "terminated: Tensor<1; Float>",
        ],
        prvt=["state: Tensor<1; Float>"],
    )

    print("\nSimpleEnv reactive module:")
    print(env._module)

    return env


def test_simpleenv_conversion():
    return simpleenv()


if __name__ == "__main__":
    reset_ctx()
    qnet = qnetwork()
    print("\n" + "="*60 + "\n")
    env = test_simpleenv_conversion()
    print("\n" + "="*60 + "\n")
    composed = Module.parallel(qnet._module, env._module)
    print(composed)