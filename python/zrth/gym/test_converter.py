from . import Context, QNetwork, SimpleEnv


def test_qnetwork_conversion():    
    ctx = Context()
    qnet = QNetwork(
        names={'extl': ['observation'], 'intf': ['q_values'], 'prvt': []},
        state_size=1,
        action_size=2,
        ctx=ctx
    )
    
    print(f"QNetwork reactive module:\n{qnet._reactive_module}")
    print(ctx)
    print(f"\nWires:")
    print(f"  qnet.intf = {qnet.intf}")
    print(f"  qnet.extl = {qnet.extl}")


def test_simpleenv_conversion():
    ctx = Context()
    env = SimpleEnv(
        names={'extl': ['q_values'], 'intf': ['observation', 'reward', 'terminated'], 'prvt': ['state']},
        ctx=ctx
    )
    
    print(f"\nSimpleEnv reactive module: {env._reactive_module}")
    print(f"\nContext:")
    print(ctx)
    print(f"\nWire access:")
    print(f"  env.intf = {env.intf}")
    print(f"  env.extl = {env.extl}")
    print(f"  env.prvt = {env.prvt}")


if __name__ == '__main__':
    test_qnetwork_conversion()
    # print("\n" + "="*60 + "\n")
    # test_simpleenv_conversion()