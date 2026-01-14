# RL to Reactive Modules Converter

This package provides automatic conversion from Python reinforcement learning code (PyTorch networks and Gymnasium environments) to reactive modules that can be composed, verified, and synthesized.

## Overview

The converter analyzes Python code and generates reactive module representations with:
- **Wire declarations**: Typed signals that connect modules
- **Terms**: Computational operations over wires
- **Modules**: Combinatorial or sequential reactive components

## Files

### Core Components

#### `backend.py`
**Purpose**: Backend abstraction layer for Rust bindings

**Design choices**:
- Automatically detects if real Rust bindings (`zrth._zrth.torch`) are available
- Falls back to mock implementations for testing without Rust
- Provides unified interface: `Wire`, `Term`, `Module`, `IType`, `DType`
- Mock implementations mirror expected Rust API

**Key classes**:
- `Wire(dtype, id)`: Represents a typed wire with unique ID
- `Term(itype, inputs, outputs)`: Represents a computational operation
- `Module`: Factory methods for `combinatorial()`, `sequential()`, `parallel()`
- `IType`: Instruction types (Add, MatMul, Ite, Argmax, etc.)
- `DType`: Data types (Tensor, Bool, None)

**Helper functions**:
- `create_wire(dtype_str, id)`: String-to-DType conversion
- `create_const(value)`: Wrap constant as IType.Const
- `create_term(itype_str, inputs, outputs)`: String-to-IType conversion

**Status**: ✅ Complete, ready for real Rust bindings

---

#### `converter.py`
**Purpose**: Automatic conversion from Python objects to reactive Modules

**Design choices**:
- **Generic, not hardcoded**: Works with any PyTorch network or Gym environment
- **Two-phase approach**: 
  1. PyTorch: Trace computation graph
  2. Gym: Parse AST of step() method
- **Automatic wire discovery**: Extracts inputs/outputs from code structure
- **Wire pairing**: Creates (latched, next) pairs automatically
- **Temporary wires**: Separate management for internal computations

**Main function**:
- `convert_to_module(ctx, python_object)`: Dispatcher that detects type and converts

**PyTorch Conversion** (`_convert_torch_module`):
- Uses `torch.jit.trace()` to capture computation graph
- Detects input size from first Linear layer
- Detects output size from last Linear layer
- Parses traced graph for operations (Linear, ReLU)
- Maps Linear to: MatMul(weight, input) + Add(result, bias)
- Maps ReLU to: Ite(Gt(x, 0), x, 0)
- Supports arbitrary number of layers

**Gym Environment Conversion** (`_convert_gym_env`):
- Requires environment to inherit from `SequentialModule`
- Reads wire declarations: `extl`, `intf`, `prvt`
- Uses AST visitor pattern to parse `step()` method
- Automatically inlines simple helper methods

**AST Visitor** (`StepMethodVisitor`):
- Converts Python statements to Terms
- Supports: assignments, arithmetic, comparisons, conditionals, method calls
- Special handling for: `argmax()`, `min()`, `max()`, `.item()`
- Maps Python operators to ITypes:
  - `+, -, *, /` → Add, Sub, Mul, Div
  - `==, !=, <, <=, >, >=` → Eq, Neq, Lt, Le, Gt, Ge
  - `a if cond else b` → Ite(cond, a, b)
  - `min(a, b)` → Ite(Lt(a, b), a, b)
  - `max(a, b)` → Ite(Gt(a, b), a, b)

**Wire Manager** (`WireManager`):
- Manages unique ID assignment
- Creates named wire pairs (for declared wires)
- Creates temp wire pairs (for intermediate computations)
- Tracks both types separately

**Status**: ✅ Complete for PyTorch and basic Gym environments

---

#### `context.py`
**Purpose**: Wire registry for name-to-dtype mapping

**Design choices**:
- Simple dictionary-based registry
- Stores only `name → dtype`, not IDs
- Shared across all modules
- IDs are assigned by converter, not stored here

**API**:
- `declare_wire(name, dtype)`: Register a wire
- `has_wire(name)`: Check if wire exists
- `get_dtype(name)`: Get wire's dtype
- `all_wires()`: List all wire names
- `num_wires()`: Count wires

**Status**: ✅ Complete

---

### Module Base Classes

#### `zrth_module.py`
**Purpose**: Abstract base classes for Python-side reactive modules

**Design choices**:
- Provides structure for declaring wires
- Does NOT implement reactive semantics (that's Rust's job)
- Used for interface declaration only

**Classes**:
- `Module`: Base class with wire declaration support
- `SequentialModule`: For stateful modules (environments)
- `CombinatorialModule`: For stateless modules (neural networks)

**Wire categories**:
- `extl`: External inputs (read but not controlled)
- `intf`: Interface signals (controlled and exposed)
- `prvt`: Private state (controlled but hidden)

**Derived properties**:
- `obs = extl + intf`: Observable wires
- `ctrl = intf + prvt`: Controlled wires

**Status**: ✅ Complete

---

### Example RL Components

#### `qnetwork.py`
**Purpose**: Example Q-network (combinatorial module)

**Details**:
- Simple feedforward network: fc1 → ReLU → fc2
- Input: observation (state)
- Output: q_values (Q-value for each action)
- Inherits from both `nn.Module` and `CombinatorialModule`

**Wire declarations**:
- `extl = ['observation']`
- `intf = ['q_values']`
- `prvt = []`

**Status**: ✅ Complete, used for testing

---

#### `simple_env.py`
**Purpose**: Example 3-state chain environment (sequential module)

**Details**:
- States: 0, 1, 2 (goal at 2)
- Actions: 0 (left), 1 (right)
- Partial observability: agent sees 0 or 1 (not full state)
- Takes q_values as input, does argmax internally
- Inherits from both `gym.Env` and `SequentialModule`

**Wire declarations**:
- `extl = ['q_values']`
- `intf = ['observation', 'reward', 'terminated']`
- `prvt = ['state']`

**State transitions**:
- Right action: `state = min(state + 1, 2)`
- Left action: `state = max(state - 1, 0)`
- Reward: 1.0 if state == 2, else 0.0
- Terminated: True if state == 2

**Status**: ✅ Complete, used for testing

---

#### `agent.py`
**Purpose**: DQN training agent (not converted to reactive module)

**Details**:
- Simplified DQN without replay buffer or target network
- Used only for training, not part of reactive system
- Provides `get_q_values()` and `train()` methods

**Status**: ✅ Complete for basic training

---

#### `main.py`
**Purpose**: Training script demonstrating RL components

**Details**:
- Creates QNetwork and SimpleEnv
- Trains with epsilon-greedy exploration
- 500 episodes, prints progress every 50

**Status**: ✅ Complete, training works

---

### Tests

#### `test_converter.py`
**Purpose**: Test individual module conversions

**Tests**:
- QNetwork → Combinatorial module (6 terms)
- SimpleEnv → Sequential module (14 terms)

**Status**: ✅ Passing

---

#### `test_composition.py`
**Purpose**: Test full conversion and composition pipeline

**Demonstrates**:
1. Converting QNetwork and SimpleEnv
2. Shared context with 5 wires
3. Module composition with `Module.parallel()`
4. Closed-loop RL system

**Status**: ✅ Passing with mocks

---

### Package Files

#### `__init__.py`
**Purpose**: Package exports and imports

**Exports**: All major classes and functions

**Status**: ✅ Complete

---

## Design Decisions

### 1. Context vs Module Separation
**Decision**: Context stores names only, not IDs
- Context: Wire registry (name → dtype)
- Module: Contains actual Wire objects with IDs
- Composition: Matches by name, Rust handles ID remapping

**Rationale**: Clean separation of concerns, enables module reuse

### 2. Wire Pairing
**Decision**: Converter creates (latched, next) pairs automatically
- Every wire becomes two Wire objects
- Latched: Current value (read from)
- Next: New value (write to)

**Rationale**: Reactive semantics require explicit time steps

### 3. Backend Abstraction
**Decision**: Mock implementations for development without Rust
- Single import switch: mocks vs real bindings
- No code changes needed when Rust becomes available

**Rationale**: Enables parallel Python/Rust development

### 4. Generic Conversion
**Decision**: Analyze code automatically
- PyTorch: Trace and parse graph
- Gym: Parse AST

**Rationale**: Works with future modules without modification

### 5. Torch Backend
**Decision**: Use torch ITypes (MatMul, Ite)
- Tensor-level operations, not scalar
- DType.Tensor for all wires

**Rationale**: Better match for neural networks, simpler Terms

---

## What's Missing / TODOs

### Immediate TODOs

1. **Rust Bindings Integration**
   - ❓ Confirm `IType.Argmax` exists in torch backend
   - ❓ Clarify wire pairing semantics in Module constructors
   - ❓ Does Rust expect flat list `[w0_latched, w0_next, w1_latched, w1_next]` or pairs?
   - ❓ Test with real Rust Module.parallel()

2. **Additional PyTorch Operations**
   - Batch normalization
   - Dropout (should be removed in trace)
   - Other activation functions (Tanh, Sigmoid, etc.)
   - Convolutional layers
   - Recurrent layers

3. **Advanced Gym Features**
   - Multi-dimensional observation spaces
   - Continuous action spaces
   - More complex state update logic
   - If/else statements (not just ternary)
   - While loops / for loops

4. **Error Handling**
   - Better error messages for unsupported patterns
   - Validation of generated Terms
   - Type checking

5. **Obligations**
   - Support for safety properties
   - Constraint generation
   - Integration with verification tools

### Future Enhancements

1. **Optimization**
   - Constant folding
   - Common subexpression elimination
   - Dead code removal

2. **Visualization**
   - Graph visualization of Terms
   - Wire flow diagrams
   - Module composition diagrams

3. **More Backends**
   - Support for SMT backend (discrete state spaces)
   - Support for toy backend (simple examples)
   - Backend selection based on module types

4. **Advanced Composition**
   - Sequential composition (not just parallel)
   - Hierarchical composition
   - Module libraries

5. **Integration with Training**
   - Convert trained model → verify → deploy pipeline
   - Property-guided training
   - Counterexample-guided training

---

## Requirements for Rust

### Required Interface in `zrth._zrth.torch`

```python
# Wire class
class Wire:
    def __init__(self, dtype: DType, id: int): ...
    @property
    def dtype(self) -> DType: ...
    @property
    def id(self) -> int: ...

# DType enum
class DType:
    Tensor: DType
    Bool: DType
    None: DType  # Note: Python uses None_

# IType - must support all these variants
class IType:
    # Arithmetic
    Add: IType
    Sub: IType
    Mul: IType
    Div: IType
    MatMul: IType
    
    # Comparison
    Eq: IType
    Neq: IType
    Lt: IType
    Le: IType
    Gt: IType
    Ge: IType
    
    # Logical
    And: IType
    Or: IType
    Not: IType
    
    # Control flow
    Ite: IType  # If-then-else
    
    # Special
    Id: IType  # Identity/pass-through
    Argmax: IType  # ❓ VERIFY THIS EXISTS
    
    # Aggregation (may not be used yet)
    Sum: IType
    Prod: IType
    
    @staticmethod
    def Const(tensor: torch.Tensor) -> IType: ...

# Term class
class Term:
    def __init__(self, itype: IType, inputs: List[Wire | IType.Const], outputs: List[Wire]): ...

# Module class
class Module:
    @staticmethod
    def combinatorial(obs_wires: List[Wire], prvt_wires: List[Wire], terms: List[Term]) -> Module: ...
    
    @staticmethod
    def sequential(obs_wires: List[Wire], ctrl_wires: List[Wire], prvt_wires: List[Wire], terms: List[Term]) -> Module: ...
    
    @staticmethod
    def parallel(modules: List[Module], shared_wires: List[str]) -> Module: ...
```

### Critical Questions

1. **Argmax**: Does `IType.Argmax` exist? If not, how should we represent argmax operations?

2. **Wire Lists**: When we pass wire lists to Module constructors, should they be:
   - Option A: Flat `[latched0, next0, latched1, next1, ...]`
   - Option B: Pairs `[(latched0, next0), (latched1, next1), ...]`
   - Option C: Just latched wires, Rust manages pairs internally

3. **Shared Wires**: In `Module.parallel(modules, shared_wires)`, the `shared_wires` parameter is a list of wire **names** (strings). How does Rust:
   - Look up wires by name in each module?
   - Handle ID remapping/unification?
   - Detect wire compatibility (dtype matching)?

4. **Const Handling**: Can `IType.Const(tensor)` be used directly in Term inputs, or does it need wrapping?

---

## Testing Instructions

### With Mocks (Current)
```bash
cd python
uv run ./rl/test_converter.py      # Test individual conversions
uv run ./rl/test_composition.py    # Test full composition
uv run ./rl/main.py                 # Test training
```

### With Real Rust Bindings (Future)
1. Implement `zrth._zrth.torch` with required interface
2. Run same tests - should automatically use real bindings
3. Verify composed module can execute

---

## Example Usage

```python
from rl import Context, QNetwork, SimpleEnv
from rl.converter import convert_to_module
from rl.backend import Module

# Create shared context
ctx = Context()

# Convert components
qnet = QNetwork(state_size=1, action_size=2)
qnet_module = convert_to_module(ctx, qnet)

env = SimpleEnv()
env_module = convert_to_module(ctx, env)

# Compose
composed = Module.parallel(
    [qnet_module, env_module],
    shared_wires=['observation', 'q_values']
)

# Now `composed` is a complete reactive system:
# - Closed-loop RL agent + environment
# - Can be verified, simulated, or synthesized
```
