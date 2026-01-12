from . import Context, QNetwork, SimpleEnv

print("Testing with real zrth bindings")
print()


def test_qnetwork_conversion():
    """Test converting QNetwork to Module"""
    
    ctx = Context()
    qnet = QNetwork(
        names={'extl': ['observation'], 'intf': ['q_values'], 'prvt': []},
        state_size=1,
        action_size=2,
        ctx=ctx
    )
    
    print("Converting QNetwork...")
    print(f"\nQNetwork reactive module: {qnet._reactive_module}")
    print(f"Context after QNetwork: {ctx.num_wires()} wires")
    print(f"\nWire access:")
    print(f"  qnet.intf = {qnet.intf}")
    print(f"  qnet.extl = {qnet.extl}")


def test_simpleenv_conversion():
    """Test converting SimpleEnv to Module"""
    
    ctx = Context()
    env = SimpleEnv(
        names={'extl': ['q_values'], 'intf': ['observation', 'reward', 'terminated'], 'prvt': ['state']},
        ctx=ctx
    )
    
    print("Converting SimpleEnv...")
    print(f"\nSimpleEnv reactive module: {env._reactive_module}")
    print(f"\nContext:")
    print(ctx)
    print(f"\nWire access:")
    print(f"  env.intf = {env.intf}")
    print(f"  env.extl = {env.extl}")
    print(f"  env.prvt = {env.prvt}")


if __name__ == '__main__':
    test_qnetwork_conversion()
    print("\n" + "="*60 + "\n")
    # test_simpleenv_conversion()