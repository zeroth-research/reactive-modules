from zrth.gym import QNetwork, SimpleEnv
from zrth import reset_ctx


def qnetwork():
    ctx = reset_ctx()
    qnet = QNetwork(
        extl=["observation: Tensor<1; Float>"],
        intf=["q_values: Tensor<2; Float>"],
        state_size=1,
        action_size=2,
        hidden_size=2,
    )

    print("QNetwork reactive module:")
    print(qnet)
    print(ctx)

    return qnet


def test_qnetwork_conversion():
    _ = qnetwork()


def simpleenv():
    ctx = reset_ctx()
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
    print(env)
    print(ctx)

    return env


def test_simpleenv_conversion():
    _ = simpleenv()


if __name__ == "__main__":
    qnet = qnetwork()
    # print("\n" + "="*60 + "\n")
    # env = test_simpleenv_conversion()
    # composed = Module.parallel(qnet._reactive_module, env._reactive_module)
