from zrth.gym import QNetwork, SimpleEnv
from zrth import Context
# from zrth import Module


def qnetwork():
    ctx = Context()
    qnet = QNetwork(
        extl=["observation"],
        intf=["q_values"],
        state_size=1,
        action_size=2,
        hidden_size=2,
        ctx=ctx,
    )

    print(f"QNetwork reactive module:\n{qnet._reactive_module}")
    print(ctx)
    print("\nWires:")
    print(f"  qnet.intf = {qnet.intf}")
    print(f"  qnet.extl = {qnet.extl}")

    return qnet


def test_qnetwork_conversion():
    _ = qnetwork()


def simpleenv():
    ctx = Context()
    env = SimpleEnv(
        extl=["q_values"],
        intf=["observation", "reward", "terminated"],
        prvt=["state"],
        ctx=ctx,
    )

    print(f"\nSimpleEnv reactive module: {env._reactive_module}")
    print(ctx)
    print("\nWire access:")
    print(f"  env.intf = {env.intf}")
    print(f"  env.extl = {env.extl}")
    print(f"  env.prvt = {env.prvt}")

    return env


def test_simpleenv_conversion():
    _ = simpleenv()


if __name__ == "__main__":
    qnet = qnetwork()
    # print("\n" + "="*60 + "\n")
    # env = test_simpleenv_conversion()
    # composed = Module.parallel(qnet._reactive_module, env._reactive_module)
