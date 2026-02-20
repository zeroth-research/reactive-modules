import torch
import inspect
import ast
import textwrap
from contextlib import contextmanager
from typing import Any
from zrth import DType, Module, Wire, Term, IType


# ============================================================================
# Main Entry Point
# ============================================================================

def convert_to_module(python_object: Any):
    """Convert a Python object to a reactive Module

    Args:
        python_object: Object to convert (QNetwork, SimpleEnv, etc.)

    Returns:
        Rust Module object
    """
    if hasattr(python_object, "forward") and isinstance(python_object, torch.nn.Module):
        return _convert_torch_module(python_object)
    elif hasattr(python_object, "step") and hasattr(python_object, "reset"):
        return _convert_gym_env(python_object)
    else:
        raise ValueError(f"Don't know how to convert {type(python_object)}")

# TODO: Convert before __init__, in __new__


# ============================================================================
# PyTorch Module Conversion
# ============================================================================

def _convert_torch_module(module: torch.nn.Module):
    """Convert PyTorch module to reactive Module
    
    Args:
        module: PyTorch module with extl/intf wire declarations
        
    Returns:
        Combinatorial Module
    """
    first_layer = None
    for layer in module.children():
        if isinstance(layer, torch.nn.Linear):
            first_layer = layer
            break
    if first_layer is None:
        raise ValueError("Can't find Linear layer to determine input shape")

    input_size = first_layer.in_features
    example_input = torch.zeros(1, input_size)
    traced = torch.jit.trace(module, example_input)

    if len(module.extl) != 1 or len(module.intf) != 1:
        raise ValueError(
            f"Only single input/output modules supported. Got {len(module.extl)} inputs, {len(module.intf)} outputs"
        )

    operations = _parse_torchscript_graph(traced.graph, module)

    terms = []
    current_wire = module.extl[0].nxt().wire()

    for op in operations:
        if op["type"] == "linear":
            current_wire = _translate_linear(current_wire, op["layer"], terms)
        elif op["type"] == "relu":
            current_wire = _translate_relu(current_wire, terms)
        else:
            raise ValueError(f"Unsupported operation: {op['type']}")

    output_wire = module.intf[0].nxt().wire()
    term = Term(IType.Id(), [output_wire], [current_wire])
    terms.append(term)

    obs = [(s.wire(), s.nxt().wire()) for s in module.extl + module.intf]
    module = Module.combinatorial(assign=terms, obs=obs)

    return module


def _parse_torchscript_graph(graph, module):
    """Parse TorchScript graph to extract operation sequence

    Args:
        graph: TorchScript graph from traced module
        module: Original PyTorch module (for layer objects)

    Returns:
        List of operation dicts: [{'type': 'linear', 'layer': layer_obj}, {'type': 'relu'}, ...]
    """
    operations = []

    layer_map = {}
    for name, layer in module.named_children():
        layer_map[name] = layer

    getattr_outputs = {}

    for node in graph.nodes():
        kind = node.kind()

        if kind == "prim::GetAttr":
            layer_name = node.s("name")
            output_name = node.output().debugName()

            if layer_name in layer_map:
                getattr_outputs[output_name] = (layer_name, layer_map[layer_name])

        elif kind == "prim::CallMethod":
            method_name = node.s("name")

            if method_name == "forward":
                inputs = list(node.inputs())
                if len(inputs) >= 1:
                    layer_ref = inputs[0].debugName()

                    if layer_ref in getattr_outputs:
                        layer_name, layer_obj = getattr_outputs[layer_ref]

                        if isinstance(layer_obj, torch.nn.Linear):
                            operations.append({"type": "linear", "layer": layer_obj})
                        else:
                            raise ValueError(
                                f"Unsupported layer type: {type(layer_obj).__name__}"
                            )

        elif kind == "aten::relu":
            operations.append({"type": "relu"})

        elif kind in ["prim::Constant", "prim::ListConstruct"]:
            continue

        else:
            raise ValueError(f"Unsupported operation: {kind}")

    return operations


def _translate_linear(input_wire: Wire, layer, terms: list[Term]) -> Wire:
    """Translate PyTorch Linear layer to reactive operations

    Creates: output = input * weight + bias

    Args:
        input_wire: Input Wire
        layer: torch.nn.Linear layer
        terms: List to append Terms to

    Returns:
        Output Wire
    """
    weight_tensor = layer.weight.data.t().contiguous()
    weight_itype = IType.Tensor(weight_tensor)
    weight_wire = Wire(DType.TensorFloat(list(weight_tensor.shape)))
    weight_term = Term(weight_itype, [weight_wire], [])
    terms.append(weight_term)
    
    matmul_dtype = DType.TensorFloat([input_wire.dtype().dims()[0], weight_tensor.shape[1]])
    matmul_wire = Wire(matmul_dtype)
    matmul_term = Term(IType.MatMul(), [matmul_wire], [input_wire, weight_wire])
    terms.append(matmul_term)
    
    bias_tensor = layer.bias.data
    bias_itype = IType.Tensor(bias_tensor)
    bias_wire = Wire(DType.TensorFloat(list(bias_tensor.shape)))
    bias_term = Term(bias_itype, [bias_wire], [])
    terms.append(bias_term)
    
    output_wire = Wire(DType.TensorFloat(list(bias_tensor.shape)))
    add_term = Term(IType.Add(), [output_wire], [matmul_wire, bias_wire])
    terms.append(add_term)
    
    return output_wire


def _translate_relu(input_wire: Wire, terms: list[Term]) -> Wire:
    """Translate ReLU activation to reactive operations

    Implements: max(0, x) = Ite(Gt(x, 0), x, 0)

    Args:
        input_wire: Input Wire
        terms: List to append Terms to

    Returns:
        Output Wire
    """
    # TODO: create an IType for ReLU and use that instead of decomposing into Gt/Ite
    input_shape = input_wire.dtype().dims()
    zero_tensor = torch.zeros(input_shape)
    zero_itype = IType.Tensor(zero_tensor)
    zero_wire = Wire(DType.TensorFloat(input_shape))
    zero_term = Term(zero_itype, [zero_wire], [])
    terms.append(zero_term)
    
    gt_wire = Wire(DType.Bool)
    gt_term = Term(IType.Gt(), [gt_wire], [input_wire, zero_wire])
    terms.append(gt_term)
    
    output_wire = Wire(input_wire.dtype())
    ite_term = Term(IType.Ite(), [output_wire], [gt_wire, input_wire, zero_wire])
    terms.append(ite_term)
    
    return output_wire


# ============================================================================
# Gym Environment Conversion
# ============================================================================

def _convert_gym_env(env):
    """Convert gym environment to reactive Module
    
    Args:
        env: Environment with reset() and step() methods, and extl/intf/prvt wire declarations
        
    Returns:
        Sequential Module
    """
    if not all(hasattr(env, attr) for attr in ('extl', 'intf', 'prvt')):
        raise ValueError("Environment must declare extl, intf, prvt wires (SequentialModule)")
    
    wire_pairs = {
        sym.name: (sym.wire(), sym.nxt().wire())
        for sym in env.extl + env.intf + env.prvt
    }
    
    init_terms = _parse_method(env, env.reset, wire_pairs)
    update_terms = _parse_method(env, env.step, wire_pairs)
    
    obs = [(s.wire(), s.nxt().wire()) for s in env.extl + env.intf]
    prvt = [(s.wire(), s.nxt().wire()) for s in env.prvt]
    
    required_wires = [wire_pairs[item.name][1] for item in env.intf + env.prvt]
    _ensure_wires_assigned(init_terms, required_wires, mode='init')
    _ensure_wires_assigned(update_terms, required_wires, mode='update')
    
    return Module.sequential(init=init_terms, update=update_terms, obs=obs, prvt=prvt)


def _parse_method(env, method, wire_pairs):
    """Parse Python method and generate Terms using AST analysis
    
    Args:
        env: Environment object
        method: Method to parse (reset or step)
        wire_pairs: Dict mapping wire names to (current_wire, next_wire) tuples
        
    Returns:
        List of Terms
    """
    source = textwrap.dedent(inspect.getsource(method))
    func_def = ast.parse(source).body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected function definition")
    
    visitor = MethodVisitor(env, wire_pairs)
    visitor.temp_vars.update({
        arg.arg: wire_pairs[arg.arg][0]
        for arg in func_def.args.args
        if arg.arg != 'self' and arg.arg in wire_pairs
    })
    
    for stmt in func_def.body:
        visitor.visit(stmt)
    
    # After visiting all statements, write interface wires from temp_vars
    for item in env.intf:
        wire_name = item.name
        if wire_name in visitor.temp_vars and wire_name not in visitor.written_wires:
            result_wire = visitor.temp_vars[wire_name]
            output_wire = wire_pairs[wire_name][1]
            term = Term(IType.Id(), [output_wire], [result_wire])
            visitor.terms.append(term)
            visitor.written_wires.add(wire_name)
    
    return visitor.terms


def _ensure_wires_assigned(terms, required_wires, mode):
    """Validate all required wires are assigned
    
    Args:
        terms: List of Terms to check
        required_wires: List of Wire objects that must be written
        mode: String ('init' or 'update') for error messages
        
    Raises:
        ValueError: If any required wire is not assigned
    """
    written_ids = set()
    for term in terms:
        for wire in term.write():
            written_ids.add(wire.id())
    
    missing_ids = [w.id() for w in required_wires if w.id() not in written_ids]
    if missing_ids:
        raise ValueError(
            f"Wire(s) with id(s) {missing_ids} not assigned in {mode}() method. "
            f"Ensure all interface and private wires are written. "
            f"Written: {sorted(written_ids)}, Required: {sorted(w.id() for w in required_wires)}"
        )


class MethodVisitor(ast.NodeVisitor):
    """AST visitor to convert Python methods to reactive Terms
    
    Intermediate values are stored as Wire objects in temp_vars.
    """
    
    BINARY_OPS = {
        ast.Add: IType.Add,
        ast.Sub: IType.Sub,
        ast.Mult: IType.Mul,
        ast.Div: IType.Div,
    }
    
    COMPARE_OPS = {
        ast.Eq: IType.Eq,
        ast.NotEq: IType.Neq,
        ast.Lt: IType.Lt,
        ast.LtE: IType.Le,
        ast.Gt: IType.Gt,
        ast.GtE: IType.Ge,
    }
    
    def __init__(self, env, wire_pairs):
        self.env = env
        self.wire_pairs = wire_pairs
        self.terms = []
        self.temp_vars = {}
        self.scopes = []
        self.written_wires = set()
    
    @contextmanager
    def _scope(self, scope_name):
        """Context manager for scope tracking
        
        Ensures scope cleanup even if exceptions occur during branch processing.
        """
        self.scopes.append(scope_name)
        try:
            yield
        finally:
            self.scopes.pop()
    
    def visit_If(self, node):
        """Handle if/else with SSA: evaluate both branches, merge with Ite
        
        Ensures each wire is written exactly once by merging branches.
        """
        cond_wire = self._convert_expr(node.test)
        
        parent_scope = dict(self.temp_vars)
        
        with self._scope('if'):
            for stmt in node.body:
                self.visit(stmt)
        if_scope_after = dict(self.temp_vars)
        
        self.temp_vars = dict(parent_scope)
        
        with self._scope('else'):
            if node.orelse:
                for stmt in node.orelse:
                    self.visit(stmt)
        else_scope_after = dict(self.temp_vars)
        
        all_vars = set(if_scope_after.keys()) | set(else_scope_after.keys())
        
        for var in all_vars:
            # Resolve variable from if branch (branch →parent → wire_pairs input)
            if_wire = if_scope_after.get(var)
            if if_wire is None:
                if_wire = parent_scope.get(var)
            if if_wire is None and var in self.wire_pairs:
                if_wire = self.wire_pairs[var][0]
            
            # Resolve variable from else branch (branch → parent → wire_pairs input)
            else_wire = else_scope_after.get(var)
            if else_wire is None:
                else_wire = parent_scope.get(var)
            if else_wire is None and var in self.wire_pairs:
                else_wire = self.wire_pairs[var][0]
            
            # If values differ, merge with Ite
            if if_wire != else_wire and if_wire is not None and else_wire is not None:
                merged_wire = Wire(if_wire.dtype())
                self.terms.append(Term(IType.Ite(), [merged_wire], [cond_wire, if_wire, else_wire]))
                self.temp_vars[var] = merged_wire
                
                # If this is a state wire and we're at top level, write the merged value to output
                if var in self.wire_pairs and len(self.scopes) == 0 and var not in self.written_wires:
                    output_wire = self.wire_pairs[var][1]
                    term = Term(IType.Id(), [output_wire], [merged_wire])
                    self.terms.append(term)
                    self.written_wires.add(var)
            elif if_wire is not None:
                self.temp_vars[var] = if_wire
                
                # Only write if at top level, not already written, and is a wire pair, AND was actually written in this if (not from parent/input)
                if var in self.wire_pairs and len(self.scopes) == 0 and var not in parent_scope and var not in self.written_wires:
                    output_wire = self.wire_pairs[var][1]
                    term = Term(IType.Id(), [output_wire], [if_wire])
                    self.terms.append(term)
                    self.written_wires.add(var)
    
    def visit_Assign(self, node):
        """Handle variable assignment"""
        if len(node.targets) != 1:
            raise ValueError("Only single assignment supported")
        
        target = node.targets[0]
        
        if isinstance(target, ast.Name):
            var_name = target.id
            result_wire = self._convert_expr(node.value)
            self.temp_vars[var_name] = result_wire
            
        elif isinstance(target, ast.Attribute) and target.attr in self.wire_pairs:
            wire_name = target.attr
            target_dtype = self.wire_pairs[wire_name][1].dtype()
            result_wire = self._convert_expr(node.value, target_dtype=target_dtype)
            self.temp_vars[wire_name] = result_wire
            
            if len(self.scopes) == 0:
                # Not in conditional - write directly to output wire
                output_wire = self.wire_pairs[wire_name][1]
                term = Term(IType.Id(), [output_wire], [result_wire])
                self.terms.append(term)
                self.written_wires.add(wire_name)
    
    def visit_AugAssign(self, node):
        """Handle augmented assignment (+=, -=, *=, /=)
        
        Expands augmented assignment to regular assignment with binary operation.
        Example: self.x += 5 becomes self.x = self.x + 5
        """
        if isinstance(node.target, ast.Attribute) and node.target.attr in self.wire_pairs:
            wire_name = node.target.attr
            
            if wire_name in self.temp_vars:
                left_wire = self.temp_vars[wire_name]
            else:
                left_wire = self.wire_pairs[wire_name][0]
            
            target_dtype = self.wire_pairs[wire_name][1].dtype()
            right_wire = self._convert_expr(node.value, target_dtype=target_dtype)
            
            op_type = type(node.op)
            if op_type not in self.BINARY_OPS:
                raise ValueError(f"Unsupported augmented assignment operator: {op_type.__name__}")
            
            result_wire = Wire(left_wire.dtype())
            itype_cls = self.BINARY_OPS[op_type]
            self.terms.append(Term(itype_cls(), [result_wire], [left_wire, right_wire]))
            
            self.temp_vars[wire_name] = result_wire
            
            if len(self.scopes) == 0:
                output_wire = self.wire_pairs[wire_name][1]
                term = Term(IType.Id(), [output_wire], [result_wire])
                self.terms.append(term)
                self.written_wires.add(wire_name)
        else:
            raise ValueError("Augmented assignment only supported for self.attribute wires")
    
    def visit_Return(self, node):
        """Handle return statement
        
        Stores return values in temp_vars. Actual wire writes happen in _parse_method.
        """
        # Parse return values into temp_vars
        if node.value is not None:
            if isinstance(node.value, ast.Tuple):
                if len(node.value.elts) != len(self.env.intf):
                    raise ValueError(
                        f"Return tuple length ({len(node.value.elts)}) "
                        f"doesn't match interface wires ({len(self.env.intf)})"
                    )
                for wire_sym, value_node in zip(self.env.intf, node.value.elts):
                    wire_name = wire_sym.name
                    target_dtype = self.wire_pairs[wire_name][1].dtype()
                    self.temp_vars[wire_name] = self._convert_expr(value_node, target_dtype=target_dtype)
            else:
                if len(self.env.intf) != 1:
                    raise ValueError(
                        f"Single return value but {len(self.env.intf)} interface wires"
                    )
                wire_name = self.env.intf[0].name
                target_dtype = self.wire_pairs[wire_name][1].dtype()
                self.temp_vars[wire_name] = self._convert_expr(node.value, target_dtype=target_dtype)
    
    def _convert_expr(self, expr, target_dtype=None):
        """Convert AST expression node to Wire object
        
        Args:
            expr: AST expression node
            target_dtype: Optional target dtype for type propagation (used for constants)
        """
        if isinstance(expr, ast.Call):
            return self._convert_call(expr, target_dtype=target_dtype)
        elif isinstance(expr, ast.BinOp):
            return self._convert_binop(expr, target_dtype=target_dtype)
        elif isinstance(expr, ast.UnaryOp):
            return self._convert_unaryop(expr, target_dtype=target_dtype)
        elif isinstance(expr, ast.BoolOp):
            return self._convert_boolop(expr)
        elif isinstance(expr, ast.Compare):
            return self._convert_compare(expr)
        elif isinstance(expr, ast.IfExp):
            return self._convert_ifexp(expr, target_dtype=target_dtype)
        elif isinstance(expr, ast.Attribute):
            return self._convert_attribute(expr)
        elif isinstance(expr, ast.Name):
            return self._convert_name(expr)
        elif isinstance(expr, ast.Constant):
            return self._convert_constant(expr, target_dtype)
        else:
            raise ValueError(f"Unsupported expression type: {type(expr).__name__}")
    
    def _convert_call(self, call, target_dtype=None):
        """Convert method/function call"""
        if isinstance(call.func, ast.Attribute):
            method = call.func.attr
            obj = call.func.value
            
            # Handle module/name-based calls first (np.*, self.*)
            # TODO: This only handles NumPy common aliases. For arbitrary aliases (import numpy as mynp),
            # would need to track imports and resolve module names.
            if isinstance(obj, ast.Name):
                name = obj.id
                if name in ('np', 'numpy'):
                    return self._convert_numpy_creation(method, call.args, target_dtype=target_dtype)
                elif name == 'self':
                    return self._inline_method(method, call.args)
            
            # Handle generic methods that work on any object
            if method == 'argmax':
                obj_wire = self._convert_expr(obj)
                result = Wire(DType.TensorFloat([1]))
                self.terms.append(Term(IType.Argmax(), [result], [obj_wire]))
                return result
            elif method == 'item':
                return self._convert_expr(obj)
            else:
                raise ValueError(f"Unsupported method: {method}")
                
        elif isinstance(call.func, ast.Name):
            func_name = call.func.id
            
            if func_name in ('min', 'max'):
                return self._convert_minmax(call.args, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")
        else:
            raise ValueError(f"Unsupported call type: {type(call.func).__name__}")
    
    def _convert_minmax(self, args, func_name):
        """Convert min/max to conditional: min(a,b) -> Ite(Lt(a,b), a, b)"""
        if len(args) != 2:
            raise ValueError(f"{func_name}() requires 2 arguments")
        
        a_wire = self._convert_expr(args[0])
        b_wire = self._convert_expr(args[1])
        
        cmp_type = IType.Lt() if func_name == 'min' else IType.Gt()
        cmp_wire = Wire(DType.Bool)
        self.terms.append(Term(cmp_type, [cmp_wire], [a_wire, b_wire]))
        
        result = Wire(a_wire.dtype())
        self.terms.append(Term(IType.Ite(), [result], [cmp_wire, a_wire, b_wire]))
        return result
    
    def _convert_numpy_creation(self, func_name, args, target_dtype=None):
        """Convert NumPy array creation to torch equivalent.

        Supports compile-time conversion of:
        - np.zeros(shape) -> Wire with Tensor term
        - np.ones(shape) -> Wire with Tensor term
        - np.array(data) -> Wire with Tensor term
        
        Shape and data must be Python literals (evaluated at conversion time).
        Respects target_dtype for creating TensorInt/Float/Bool arrays.
        """
        if func_name == 'zeros':
            if len(args) != 1:
                raise ValueError(f"np.zeros() requires 1 argument, got {len(args)}")
            shape = self._eval_shape(args[0])
            
            if target_dtype and target_dtype.kind() == "TensorInt":
                tensor_data = torch.zeros(*shape, dtype=torch.long)
            elif target_dtype and target_dtype.kind() == "TensorBool":
                tensor_data = torch.zeros(*shape, dtype=torch.bool)
            else:
                tensor_data = torch.zeros(*shape, dtype=torch.float32)
            
        elif func_name == 'ones':
            if len(args) != 1:
                raise ValueError(f"np.ones() requires 1 argument, got {len(args)}")
            shape = self._eval_shape(args[0])
            
            if target_dtype and target_dtype.kind() == "TensorInt":
                tensor_data = torch.ones(*shape, dtype=torch.long)
            elif target_dtype and target_dtype.kind() == "TensorBool":
                tensor_data = torch.ones(*shape, dtype=torch.bool)
            else:
                tensor_data = torch.ones(*shape, dtype=torch.float32)
            
        elif func_name == 'array':
            if len(args) != 1:
                raise ValueError(f"np.array() requires 1 argument, got {len(args)}")
            data = self._eval_literal(args[0])
            
            if target_dtype and target_dtype.kind() == "TensorInt":
                tensor_data = torch.tensor(data, dtype=torch.long)
            elif target_dtype and target_dtype.kind() == "TensorBool":
                tensor_data = torch.tensor(data, dtype=torch.bool)
            else:
                tensor_data = torch.tensor(data, dtype=torch.float32)
        
        else:
            raise ValueError(f"Unsupported NumPy function: np.{func_name}()")
        
        shape = list(tensor_data.size())
        if target_dtype:
            dtype = target_dtype.reshape(shape)
        else:
            dtype = DType.TensorFloat(shape)
        
        const_wire = Wire(dtype)
        self.terms.append(Term(IType.Tensor(tensor_data), [const_wire]))
        return const_wire
    
    def _eval_shape(self, shape_node):
        """Evaluate shape argument at compile time.
        
        Accepts:
        - Single int: np.zeros(5) -> (5,)
        - Tuple: np.zeros((3, 3)) -> (3, 3)
        - List: np.zeros([10]) -> (10,)
        
        Returns:
            Tuple of ints representing the shape
        """
        if isinstance(shape_node, ast.Constant):
            return (shape_node.value,)
        elif isinstance(shape_node, (ast.Tuple, ast.List)):
            shape = []
            for elt in shape_node.elts:
                if not isinstance(elt, ast.Constant):
                    raise ValueError(f"Shape must be constant, got {type(elt).__name__}")
                shape.append(elt.value)
            return tuple(shape)
        else:
            raise ValueError(f"Shape must be int or tuple, got {type(shape_node).__name__}")
    
    def _eval_literal(self, data_node):
        """Recursively evaluate Python literal for np.array() data.
        
        Supports nested lists/tuples of constants only.
        Examples: [1, 2, 3], [[1, 2], [3, 4]], (1.0, 2.0, 3.0)
        
        Returns:
            Python list/scalar that can be passed to torch.tensor()
        """
        if isinstance(data_node, ast.Constant):
            return data_node.value
        elif isinstance(data_node, (ast.List, ast.Tuple)):
            return [self._eval_literal(elt) for elt in data_node.elts]
        else:
            raise ValueError(f"np.array() data must be literal, got {type(data_node).__name__}")
    
    
    def _convert_binop(self, binop, target_dtype=None):
        """Convert binary operation
        
        Dtype propagation:
        - With target_dtype: both operands inherit it (e.g., self.x = 2 + 3 → both TensorInt)
        - Without: right operand inherits from left (e.g., self.x + 3 → 3 matches x's dtype)
        """
        if target_dtype:
            # Top-down propagation: assignment target determines operand types
            left_wire = self._convert_expr(binop.left, target_dtype=target_dtype)
            right_wire = self._convert_expr(binop.right, target_dtype=target_dtype)
            result_dtype = target_dtype
        else:
            # Bottom-up inference: left operand determines right
            left_wire = self._convert_expr(binop.left)
            right_wire = self._convert_expr(binop.right, target_dtype=left_wire.dtype())
            result_dtype = left_wire.dtype()
        
        op_type = type(binop.op)
        if op_type not in self.BINARY_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        
        result = Wire(result_dtype)
        itype_cls = self.BINARY_OPS[op_type]
        self.terms.append(Term(itype_cls(), [result], [left_wire, right_wire]))
        return result
    
    def _convert_unaryop(self, unaryop, target_dtype=None):
        """Convert unary operation (not, -, +)
        
        - not x: Always returns Bool
        - -x: Propagates target_dtype to operand and zero constant
        - +x: No-op, returns operand with target_dtype propagated
        """
        op_type = type(unaryop.op)
        if op_type == ast.Not:
            # not x -> Ite(x, False, True) - always Bool
            operand_wire = self._convert_expr(unaryop.operand)
            false_wire = self._convert_constant(ast.Constant(False))
            true_wire = self._convert_constant(ast.Constant(True))
            result = Wire(DType.Bool)
            self.terms.append(Term(IType.Ite(), [result], [operand_wire, false_wire, true_wire]))
            return result
        elif op_type == ast.USub:
            # -x -> 0 - x
            # If target_dtype provided, use it for the constant and result
            if target_dtype:
                operand_wire = self._convert_expr(unaryop.operand, target_dtype=target_dtype)
                zero_wire = self._convert_constant(ast.Constant(0), target_dtype=target_dtype)
                result_dtype = target_dtype
            else:
                operand_wire = self._convert_expr(unaryop.operand)
                zero_wire = self._convert_constant(ast.Constant(0), target_dtype=operand_wire.dtype())
                result_dtype = operand_wire.dtype()
            result = Wire(result_dtype)
            self.terms.append(Term(IType.Sub(), [result], [zero_wire, operand_wire]))
            return result
        elif op_type == ast.UAdd:
            return self._convert_expr(unaryop.operand, target_dtype=target_dtype)
        else:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    
    def _convert_boolop(self, boolop):
        """Convert boolean operation (and/or) to nested Ite expressions
        
        Python: a and b -> Ite(a, b, False)
        Python: a or b  -> Ite(a, True, b)
        """
        if len(boolop.values) < 2:
            raise ValueError("BoolOp must have at least 2 operands")
        
        wires = [self._convert_expr(val) for val in boolop.values]
        is_and = isinstance(boolop.op, ast.And)
        
        if not is_and and not isinstance(boolop.op, ast.Or):
            raise ValueError(f"Unsupported boolean operator: {type(boolop.op).__name__}")
        
        # Build nested Ite from right to left
        result = wires[-1]
        for wire in reversed(wires[:-1]):
            false_wire = self._convert_constant(ast.Constant(False))
            true_wire = self._convert_constant(ast.Constant(True))
            
            merged = Wire(DType.Bool)
            if is_and:
                # a and b -> Ite(a, b, False)
                self.terms.append(Term(IType.Ite(), [merged], [wire, result, false_wire]))
            else:
                # a or b -> Ite(a, True, b)
                self.terms.append(Term(IType.Ite(), [merged], [wire, true_wire, result]))
            result = merged
        return result
    
    def _convert_compare(self, compare):
        """Convert comparison operation
        
        Handles both simple comparisons (a < b) and chains (a < b < c).
        Chains are expanded: a < b < c becomes (a < b) and (b < c)
        """
        comparison_wires = []
        left = compare.left
        
        for op, comparator in zip(compare.ops, compare.comparators):
            left_wire = self._convert_expr(left)
            right_wire = self._convert_expr(comparator, target_dtype=left_wire.dtype())
            
            op_type = type(op)
            if op_type not in self.COMPARE_OPS:
                raise ValueError(f"Unsupported comparison operator: {op_type.__name__}")
            
            cmp_wire = Wire(DType.Bool)
            itype_cls = self.COMPARE_OPS[op_type]
            self.terms.append(Term(itype_cls(), [cmp_wire], [left_wire, right_wire]))
            comparison_wires.append(cmp_wire)
            left = comparator
        
        # Combine with AND (single comparison returns directly)
        result = comparison_wires[0]
        for comp_wire in comparison_wires[1:]:
            false_wire = self._convert_constant(ast.Constant(False))
            merged = Wire(DType.Bool)
            # result and comp_wire -> Ite(result, comp_wire, False)
            self.terms.append(Term(IType.Ite(), [merged], [result, comp_wire, false_wire]))
            result = merged
        
        return result
    
    def _convert_ifexp(self, ifexp, target_dtype=None):
        """Convert ternary conditional to Ite
        
        Propagates target_dtype to both branches (e.g., self.x = 5 if c else 10 → both TensorInt)
        """
        cond_wire = self._convert_expr(ifexp.test)
        true_wire = self._convert_expr(ifexp.body, target_dtype=target_dtype)
        false_wire = self._convert_expr(ifexp.orelse, target_dtype=target_dtype)
        
        result_dtype = target_dtype if target_dtype else true_wire.dtype()
        result = Wire(result_dtype)
        self.terms.append(Term(IType.Ite(), [result], [cond_wire, true_wire, false_wire]))
        return result
    
    def _convert_name(self, name):
        """Convert variable reference"""
        var_name = name.id
        if var_name in self.temp_vars:
            return self.temp_vars[var_name]
        else:
            raise ValueError(f"Unknown variable: {var_name}")
    
    def _convert_constant(self, constant, target_dtype=None):
        """Convert constant literal to Wire with Tensor term
        
        Args:
            constant: AST Constant node
            target_dtype: Optional dtype (inferred from context). Defaults to TensorFloat if None.
        """
        value = constant.value
        
        if isinstance(value, bool):
            tensor_data = torch.tensor([value])
            dtype = DType.TensorBool([1])
        
        elif isinstance(value, (int, float)):
            if target_dtype is None:
                # TODO: Consider inferring dtype from value (int vs float) or raising error instead of defaulting
                target_dtype = DType.TensorFloat([])
                tensor_data = torch.tensor([float(value)], dtype=torch.float32)
            elif target_dtype.kind() == "TensorInt":
                tensor_data = torch.tensor([int(value)], dtype=torch.long)
            elif target_dtype.kind() == "TensorFloat":
                tensor_data = torch.tensor([float(value)], dtype=torch.float32)
            elif target_dtype.kind() == "TensorBool":
                tensor_data = torch.tensor([bool(value)])
            else:
                raise ValueError(f"Unsupported target dtype kind: {target_dtype.kind()}")
            
            dtype = target_dtype.reshape([1])
        
        elif isinstance(value, (list, tuple, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                tensor_data = value
            else:
                if target_dtype and target_dtype.kind() == "TensorInt":
                    tensor_data = torch.tensor(value, dtype=torch.long)
                elif target_dtype and target_dtype.kind() == "TensorBool":
                    tensor_data = torch.tensor(value, dtype=torch.bool)
                else:
                    tensor_data = torch.tensor(value, dtype=torch.float32)
            
            if target_dtype:
                dtype = target_dtype.reshape(list(tensor_data.size()))
            else:
                dtype = DType.TensorFloat(list(tensor_data.size()))
        
        else:
            raise ValueError(f"Unsupported constant type: {type(value)}")
        
        const_wire = Wire(dtype)
        self.terms.append(Term(IType.Tensor(tensor_data), [const_wire]))
        return const_wire
    
    def _convert_attribute(self, attr):
        """Convert attribute access: self.state, self.observation, etc.
        
        First checks temp_vars for locally-assigned values (within this method).
        Falls back to latched wire (reading state from previous cycle).
        """
        if isinstance(attr.value, ast.Name) and attr.value.id == 'self':
            wire_name = attr.attr
            
            if wire_name in self.temp_vars:
                return self.temp_vars[wire_name]
            
            if wire_name in self.wire_pairs:
                return self.wire_pairs[wire_name][0]
            else:
                raise ValueError(f"Unknown wire: {wire_name}")
        else:
            raise ValueError(f"Unsupported attribute access: {ast.unparse(attr)}")
    
    def _inline_method(self, method_name, args):
        """Inline simple method by extracting its return expression"""
        if not hasattr(self.env, method_name):
            raise ValueError(f"Method not found: {method_name}")
        
        method = getattr(self.env, method_name)
        source = textwrap.dedent(inspect.getsource(method))
        method_def = ast.parse(source).body[0]
        if not isinstance(method_def, ast.FunctionDef):
            raise ValueError(f"Expected function definition for {method_name}")
        
        return_stmt = next((stmt for stmt in method_def.body if isinstance(stmt, ast.Return)), None)
        if return_stmt is None or return_stmt.value is None:
            raise ValueError(f"Method {method_name} has no return value")
        
        return self._convert_expr(return_stmt.value)
