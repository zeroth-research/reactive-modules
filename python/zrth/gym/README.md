# Gym Converter

This module converts Python gymnasium environments and PyTorch neural networks into reactive dataflow modules. The converter enables compositional reasoning about agent-environment interactions by transforming imperative Python code into a functional reactive representation.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Concepts](#key-concepts)
- [How It Works](#how-it-works)
- [Static Single Assignment (SSA)](#static-single-assignment-ssa)
- [Limitations](#limitations)
- [Design Decisions](#design-decisions)
- [Future Enhancements](#future-enhancements)
- [Examples](#examples)

## Overview

The converter transforms:
- **Gymnasium environments** (with `reset()` and `step()` methods) → Sequential reactive modules
- **PyTorch neural networks** (with `forward()` method) → Combinatorial reactive modules

These modules can then be composed together to create complete agent-environment systems that are amenable to formal analysis, verification, and symbolic reasoning.

The converter uses a **global context pattern** where all functions access the reactive module context via `get_ctx()`, eliminating the need to pass context parameters explicitly.

## Architecture

### Core Components

1. **Wire Pairs**: Represent synchronous state boundaries
   - `(latched, next)` - read current value, write next value
   - Used for: interface outputs, external inputs, private state

2. **Terms**: Atomic dataflow operations
   - `IType` operations: Add, Sub, Mul, Eq, Lt, Ite, Argmax, etc.
   - Connect input wires to output wires
   - Each wire written exactly once (SSA property)

3. **Modules**: Collections of Terms with wire interfaces
   - `Sequential`: Has init and update phases (for stateful environments)
   - `Combinatorial`: Single assignment phase (for stateless networks)

### Conversion Pipeline

```
Python Source Code
    ↓
AST Parsing (inspect + ast modules)
    ↓
MethodVisitor (AST Visitor Pattern)
    ↓
Term Generation (with SSA transform)
    ↓
Wire Assignment Validation
    ↓
Reactive Module (via Module.sequential or Module.combinatorial)
```

**Global Context**: All functions use `get_ctx()` to access the reactive module context. This is initialized once with `set_ctx(Context())` at the start of your program.

## Key Concepts

### Wire Semantics

**Single Wires** (temporary computation):
```python
temp = x + 1
```
→ Single wire `w10` for temporary value

**Wire Pairs** (state/interface boundaries):
```python
self.state = self.state + 1
```
→ Pair `(state_l, state_n)`:
- `state_l`: Read current state (latched)
- `state_n`: Write next state

### Term Structure

Each Term represents a single operation:
```
IType output_wire; input_wire1, input_wire2
```

Example:
```
Add w10; w8, w9     # w10 = w8 + w9
Ite w15; w12, w13, w14   # w15 = w12 ? w13 : w14
```

### Sequential vs Combinatorial

**Sequential Module** (environment):
- **init**: Reset behavior - initializes all state
- **update**: Step behavior - updates state based on inputs
- Has private state that persists across steps

**Combinatorial Module** (neural network):
- Single **assign** phase - pure function from input to output
- No state, just computation

## How It Works

### 1. AST Parsing

Extract Python source code and parse into Abstract Syntax Tree:

```python
method_source = inspect.getsource(env.step)
tree = ast.parse(textwrap.dedent(method_source))
```

### 2. AST Visitor Pattern

`MethodVisitor` traverses the AST and generates Terms:

```python
class MethodVisitor(ast.NodeVisitor):
    def __init__(self, env, wire_pairs):
        self.env = env
        self.ctx = get_ctx()  # Global context access
        self.wire_pairs = wire_pairs
        self.temp_vars = {}   # SSA variable tracking
        self.scopes = []      # Conditional scope tracking
        self.written_wires = set()
    
    def visit_Assign(self, node): ...
    def visit_If(self, node): ...
    def visit_Return(self, node): ...
    def _convert_expr(self, expr): ...
```

**Supported constructs**:
- Variable assignment: `x = expr`
- State update: `self.state = expr`
- Arithmetic: `+`, `-`, `*`, `/`
- Comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Conditionals: `if/else` statements with SSA merging, ternary `a if cond else b`
- Method calls: `.argmax()`, `.item()`, `min()`, `max()`
- Method inlining: Simple helper methods with single return

### 3. Wire Tracking

**temp_vars**: Maps variable names to current wire IDs
```python
self.temp_vars = {
    'action': w17,
    'state': w30,
    'reward': w35
}
```

Reading a variable looks up its wire in `temp_vars`.

### 4. Expression Conversion

Every expression converts to a wire ID:

```python
# self.state + 1
state_wire = self.temp_vars['state']  # w8
const_wire = create_constant(1)        # w20
result_wire = create_add(state_wire, const_wire)  # w21
```

Generates Terms:
```
Const(tensor) w20;
Add w21; w8, w20
```

## Static Single Assignment (SSA)

SSA is the **key technique** that enables correct handling of control flow, particularly if/else statements.

### The Problem

Without SSA, both branches of an if/else would write to the same output wire:

```python
if action == 1:
    self.state = self.state + 1  # writes to state_n
else:
    self.state = self.state - 1  # writes to state_n again!
```

This causes a "write after write" error because each wire must be written exactly once.

### The SSA Solution

SSA ensures each wire is written exactly once by:

1. **Separate Scopes**: Each branch computes in isolation
   - Track scope depth with `self.scopes` stack
   - Assignments inside branches only update `temp_vars`, don't write to output wires

2. **Branch Isolation**: Both branches evaluate independently
   - If-branch: `self.state = ...` → updates `temp_vars['state']` to `w24`
   - Else-branch: `self.state = ...` → updates `temp_vars['state']` to `w29`

3. **Explicit Merging**: At convergence point, use Ite to merge
   ```
   Ite w30; w19, w24, w29    # w30 = condition ? w24 : w29
   Id w9; w30                 # Write merged result to output
   ```

### SSA Example

Python code:
```python
def step(self, q_values):
    action = q_values.argmax()
    if action == 1:
        self.state = min(self.state + 1, 2)
    else:
        self.state = max(self.state - 1, 0)
    return self.state
```

Generated Terms (simplified):
```
Argmax w17; w0                    # action = q_values.argmax()
Eq w19; w17, 1                    # action == 1

# If-branch: compute state + 1, clamped to 2
Add w21; w8, 1
Lt w23; w21, 2
Ite w24; w23, w21, 2              # min(state+1, 2)

# Else-branch: compute state - 1, clamped to 0
Sub w26; w8, 1
Gt w28; w26, 0
Ite w29; w28, w26, 0              # max(state-1, 0)

# Merge with SSA
Ite w30; w19, w24, w29            # SSA merge: condition ? if_value : else_value
Id w9; w30                        # Write to state_n output wire (once!)
Id w3; w30                        # Return value -> observation_n
```

## Limitations

### Current Limitations

1. **No Loops**
   - Cannot handle `for` or `while` loops
   - Reason: Unbounded iteration doesn't map to finite dataflow graph
   
2. **No Complex Data Structures**
   - Arrays/lists: Only simple tensor operations
   - Reason: Reactive modules operate on atomic data types (Tensor, Bool)

3. **Limited Control Flow**
   - If/else: ✅ Supported with SSA
   - Nested if/else: ✅ Supported
   - Match statements: ❌ Not yet implemented
   - Try/except: ❌ Not supported (no exceptions in dataflow)

4. **Method Call Restrictions**
   - Only simple helper methods can be inlined
   - No recursive calls
   - No side effects (besides state updates)

### Why These Limitations Exist

The reactive module formalism is designed for:
- **Finite, bounded computation**: Every module has fixed size
- **Synchronous execution**: Init and update phases execute atomically
- **Compositional reasoning**: Modules compose algebraically

These properties are incompatible with:
- Unbounded loops (unless unrolled)
- Mutable complex data structures
- Arbitrary side effects

## Design Decisions

### Why AST Over Alternatives?

I evaluated three approaches:

#### 1. AST Parsing ✅ (Chosen)

**Pros**:
- Clean, high-level representation
- Easy to debug (matches source code)
- Good error messages with line numbers
- Full type information available
- Can implement SSA at AST level

**Cons**:
- Requires parsing source code
- More verbose to implement

**Verdict**: Best balance of clarity and capability.

#### 2. Bytecode Analysis ❌

**Pros**:
- No source code needed
- Closer to actual execution

**Cons**:
- Harder to debug (obscure instructions)
- Worse error messages (no line numbers)
- More complex SSA implementation
- Still can't handle loops without unrolling

**Verdict**: No significant advantage, much harder to use.

#### 3. Custom DSL ❌

**Pros**:
- Complete control over semantics
- Easy to implement

**Cons**:
- Users must learn new language
- No IDE support
- Poor Python interop
- Defeats purpose of Python-based development

**Verdict**: Too much friction for users.

## Code Organization

The converter is organized into three main sections:

1. **Main Entry Point**
   - `convert_to_module()`: Dispatch based on object type

2. **PyTorch Module Conversion**
   - TorchScript graph parsing
   - Direct Sym wire access (no parameter parsing needed)
   - Linear layer and ReLU activation translation

3. **Gym Environment Conversion**
   - Method parsing with AST
   - Wire pair creation from Sym objects  
   - Wire validation
   - MethodVisitor with SSA for control flow

## Future Enhancements

### 1. Loop Unrolling (ROI: Medium)

**Approach**:
```python
for i in range(3):
    x = x + 1
```

Unroll to:
```python
x = x + 1
x = x + 1
x = x + 1
```

**Limitations**:
- Only constant bounds
- Must be unrollable (no dynamic iteration)

**Use case**: Fixed-size arrays, matrix operations

### 2. Control Flow Graph (ROI: Low)

**Approach**: Build CFG, insert phi nodes at merge points

**Limitations**:
- Complex implementation
- Still can't handle unbounded loops
- SSA already handles most cases

**Verdict**: Not worth it for current use cases.

### 3. Match Statement Support (ROI: High)

**Approach**: Convert match to nested if/elif/else

**Benefit**: Better pattern matching support

**Verdict**: Easy win if needed.

### 4. Array/Tensor Operations (ROI: High)

**Approach**: Add IType operations for indexing, slicing

**Benefit**: More expressive computations

**Limitation**: Fixed-size tensors only

**Verdict**: High value if use cases emerge.
