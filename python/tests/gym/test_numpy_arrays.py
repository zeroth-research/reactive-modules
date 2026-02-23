"""Test NumPy array creation support in converter."""

import numpy as np
from zrth.gym.zrth_module import Module


class ArrayEnv(Module):
    """Test environment using NumPy array creation."""

    def __init__(self):
        super().__init__(
            extl=["action: Tensor<1; Int>"],
            intf=["done: Tensor<1; Bool>"],
            prvt=[
                "grid: Tensor<3, 3; Float>",
                "weights: Tensor<5; Float>",
                "values: Tensor<3; Float>",
                "matrix: Tensor<2, 2; Float>",
            ],
        )

        self.grid = np.zeros((3, 3))
        self.weights = np.ones(5)
        self.values = np.array([1.0, 2.0, 3.0])
        self.matrix = np.array([[1.0, 2.0], [3.0, 4.0]])

    def reset(self):
        self.grid = np.zeros((3, 3))
        self.weights = np.ones(5)
        self.values = np.array([1.0, 2.0, 3.0])
        self.matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        done = False
        return done

    def step(self, action):
        self.grid = self.grid
        self.weights = self.weights
        self.values = self.values
        self.matrix = self.matrix
        done = action > 10
        return done


def test_numpy_zeros():
    """Test np.zeros() conversion."""
    module = ArrayEnv()

    # Verify the module was created
    assert module is not None

    # Get the init transition and check it
    init_trans = module.init_as_transition()
    init_str = str(init_trans.intf_out())

    # Check that grid wire exists with correct shape
    assert "Tensor<3, 3; Float>" in init_str
    print(f"✓ np.zeros((3, 3)) -> Tensor<3, 3; Float>")


def test_numpy_ones():
    """Test np.ones() conversion."""
    module = ArrayEnv()

    init_trans = module.init_as_transition()
    init_str = str(init_trans.intf_out())

    # Check that weights wire exists with correct shape
    assert "Tensor<5; Float>" in init_str
    print(f"✓ np.ones(5) -> Tensor<5; Float>")


def test_numpy_array_1d():
    """Test np.array() with 1D list conversion."""
    module = ArrayEnv()

    init_trans = module.init_as_transition()
    init_str = str(init_trans.intf_out())

    # Check that values wire exists with correct shape
    assert "Tensor<3; Float>" in init_str
    print(f"✓ np.array([1.0, 2.0, 3.0]) -> Tensor<3; Float>")


def test_numpy_array_2d():
    """Test np.array() with 2D list conversion."""
    module = ArrayEnv()

    init_trans = module.init_as_transition()
    init_str = str(init_trans.intf_out())

    # Check that matrix wire exists with correct shape
    assert "Tensor<2, 2; Float>" in init_str
    print(f"✓ np.array([[1.0, 2.0], [3.0, 4.0]]) -> Tensor<2, 2; Float>")


if __name__ == "__main__":
    print("Testing NumPy array creation...\n")

    print("1. Testing np.zeros()...")
    test_numpy_zeros()
    print()

    print("2. Testing np.ones()...")
    test_numpy_ones()
    print()

    print("3. Testing np.array() with 1D list...")
    test_numpy_array_1d()
    print()

    print("4. Testing np.array() with 2D list...")
    test_numpy_array_2d()
    print()

    print("All NumPy array creation tests passed!")
