from .environments import SimpleEnv


def simpleenv():
    env = SimpleEnv()

    print("\nSimpleEnv reactive module:")
    print(env)

    return env


def test_simpleenv_conversion():
    m = simpleenv()
    print(m.to_lean())
