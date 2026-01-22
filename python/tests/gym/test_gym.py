from zrth.gym import QNetwork, SimpleEnv
from zrth import Context


def qnetwork():
    ctx = Context()
    qnet = QNetwork(
        extl=["observation: Tensor<1>"],
        intf=["q_values: Tensor<2>"],
        state_size=1,
        action_size=2,
        hidden_size=2,
        ctx=ctx,
    )

    print("QNetwork reactive module:")
    print(qnet)
    print(ctx)

    return qnet


def test_qnetwork_conversion():
    _ = qnetwork()


def simpleenv():
    ctx = Context()
    env = SimpleEnv(
        extl=["q_values: Tensor<2>"],
        intf=["observation: Tensor<1>", "reward: Tensor<1>", "terminated: Tensor<1>"],
        prvt=["state: Tensor<1>"],
        ctx=ctx,
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
