from rl import Context, QNetwork, SimpleEnv

print("Testing with real mylib bindings")
print()


def test_qnetwork_conversion():
    """Test converting QNetwork to Module"""
    
    ctx = Context()
    qnet = QNetwork(
        ctx,
        names={'extl': ['observation'], 'intf': ['q_values'], 'prvt': []},
        state_size=1,
        action_size=2
    )
    
    print("Converting QNetwork...")
    print(f"\nQNetwork reactive module: {qnet._reactive_module}")
    print(f"Context after QNetwork: {ctx.num_wires()} wires")
    print(f"\nWire access:")
    print(f"  qnet.intf = {qnet.intf}")
    print(f"  qnet.intf_named = {qnet.intf_named}")


def test_simpleenv_conversion():
    """Test converting SimpleEnv to Module"""
    
    ctx = Context()
    env = SimpleEnv(
        ctx,
        names={'extl': ['q_values'], 'intf': ['observation', 'reward', 'terminated'], 'prvt': ['state']}
    )
    
    print("Converting SimpleEnv...")
    print(f"\nSimpleEnv reactive module: {env._reactive_module}")
    print(f"\nContext:")
    print(ctx)
    print(f"\nWire access:")
    print(f"  env.intf = {env.intf}")
    print(f"  env.intf_named = {env.intf_named}")


if __name__ == '__main__':
    test_qnetwork_conversion()
    print("\n" + "="*60 + "\n")
    # test_simpleenv_conversion()