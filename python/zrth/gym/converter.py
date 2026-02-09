import torch
import inspect
import ast
import textwrap
from typing import Any
from zrth.context import get_ctx
from zrth import DType, Module
from zrth.expr import Expr


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


# ============================================================================
# PyTorch Module Conversion
# ============================================================================

def _convert_torch_module(module: torch.nn.Module):
    """Convert PyTorch module to reactive Module using Expr API
    
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

    ctx = get_ctx()
    terms = []
    ctx.push_terms_frame(terms)

    current_expr = module.extl[0].nxt()

    for op in operations:
        if op["type"] == "linear":
            current_expr = _translate_linear(current_expr, op["layer"])
        elif op["type"] == "relu":
            current_expr = _translate_relu(current_expr)
        else:
            raise ValueError(f"Unsupported operation: {op['type']}")

    Expr("assign", current_expr, module.intf[0].nxt())
    
    ctx.pop_terms_frame()

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


def _translate_linear(input_expr, layer):
    """Translate PyTorch Linear layer to reactive operations

    Args:
        input_expr: Input Expr
        layer: torch.nn.Linear layer

    Returns:
        Output Expr (terms collected automatically via terms_frame)
    """
    weight_tensor = layer.weight.data.t().contiguous()
    weight_expr = Expr("const", weight_tensor)
    
    matmul_expr = Expr("arith.matmul", input_expr, weight_expr)
    
    bias_tensor = layer.bias.data
    bias_expr = Expr("const", bias_tensor)
    
    output_expr = Expr("arith.add", matmul_expr, bias_expr)
    
    return output_expr


def _translate_relu(input_expr):
    """Translate ReLU activation to reactive operations

    Implements: max(0, x) = Ite(Gt(x, 0), x, 0)

    Args:
        input_expr: Input Expr

    Returns:
        Output Expr (terms collected automatically via terms_frame)
    """
    input_shape = input_expr.dtype().dims()
    zero_tensor = torch.zeros(input_shape)
    zero_expr = Expr("const", zero_tensor)
    
    gt_expr = Expr("cmp.gt", input_expr, zero_expr)
    
    output_expr = Expr("ite", gt_expr, input_expr, zero_expr)
    
    return output_expr


# ============================================================================
# Gym Environment Conversion
# ============================================================================

def _convert_gym_env(env):
    """Convert gym environment to reactive Module using Expr API
    
    Args:
        env: Environment with reset() and step() methods, and extl/intf/prvt wire declarations
        
    Returns:
        Sequential Module
    """
    # Validate wire declarations
    if not all(hasattr(env, attr) for attr in ('extl', 'intf', 'prvt')):
        raise ValueError("Environment must declare extl, intf, prvt wires (SequentialModule)")
    
    syms = {sym.name: sym for sym in env.extl + env.intf + env.prvt}
    
    init_terms = _parse_method(env, env.reset, syms)
    update_terms = _parse_method(env, env.step, syms)
    
    obs = [(s.wire(), s.nxt().wire()) for s in env.extl + env.intf]
    prvt = [(s.wire(), s.nxt().wire()) for s in env.prvt]
    
    required_intf_wires = [syms[item.name].nxt().wire() for item in env.intf]
    _ensure_wires_assigned(init_terms, required_intf_wires, mode='init')
    _ensure_wires_assigned(update_terms, required_intf_wires, mode='update')
    
    return Module.sequential(init=init_terms, update=update_terms, obs=obs, prvt=prvt)


def _parse_method(env, method, syms):
    """Parse Python method and generate Terms using AST analysis and Expr API
    
    Args:
        env: Environment object
        method: Method to parse (reset or step)
        syms: Dict mapping wire names to Sym objects
        
    Returns:
        List of Terms
    """
    source = textwrap.dedent(inspect.getsource(method))
    func_def = ast.parse(source).body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected function definition")
    
    ctx = get_ctx()
    terms = []
    ctx.push_terms_frame(terms)
    
    visitor = MethodVisitor(env, syms)
    # Store Sym objects (which are Expr) in temp_vars for method arguments
    visitor.temp_vars.update({
        arg.arg: syms[arg.arg]
        for arg in func_def.args.args
        if arg.arg != 'self' and arg.arg in syms
    })
    
    for stmt in func_def.body:
        visitor.visit(stmt)
    
    ctx.pop_terms_frame()
    return terms


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
    
    for wire in required_wires:
        if wire.id() not in written_ids:
            raise ValueError(
                f"Interface wire (id={wire.id()}) was not assigned in {mode}() method. "
                f"Ensure you define a variable matching your intf declaration. "
                f"Variable names must match exactly with the interface wire names."
            )


class MethodVisitor(ast.NodeVisitor):
    """AST visitor to convert Python methods to reactive Terms using Expr API
    
    All intermediate values are stored as Expr objects in temp_vars.
    Terms are collected automatically via context's terms_frame mechanism.
    """
    
    BINARY_OPS = {
        ast.Add: 'arith.add',
        ast.Sub: 'arith.sub',
        ast.Mult: 'arith.mul',
        ast.Div: 'arith.div',
    }
    
    COMPARE_OPS = {
        ast.Eq: 'cmp.eq',
        ast.NotEq: 'cmp.neq',
        ast.Lt: 'cmp.lt',
        ast.LtE: 'cmp.le',
        ast.Gt: 'cmp.gt',
        ast.GtE: 'cmp.ge',
    }
    
    def __init__(self, env, syms):
        self.env = env
        self.syms = syms
        self.temp_vars = {}
        self.scopes = []
        self.written_wires = set()
    
    def visit_If(self, node):
        """Handle if/else with SSA: evaluate both branches, merge with Ite
        
        Ensures each wire is written exactly once by merging diverged variables
        with Ite expressions at the merge point.
        """
        cond_expr = self._convert_expr(node.test)
        
        parent_scope = dict(self.temp_vars)
        
        self.scopes.append('if')
        for stmt in node.body:
            self.visit(stmt)
        if_scope_after = dict(self.temp_vars)
        self.scopes.pop()
        
        self.temp_vars = dict(parent_scope)
        
        self.scopes.append('else')
        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
        else_scope_after = dict(self.temp_vars)
        self.scopes.pop()
        
        all_vars = set(if_scope_after.keys()) | set(else_scope_after.keys())
        
        for var in all_vars:
            if_expr = if_scope_after.get(var, parent_scope.get(var))
            else_expr = else_scope_after.get(var, parent_scope.get(var))
            
            if if_expr != else_expr:
                merged_expr = Expr("ite", cond_expr, if_expr, else_expr)
                self.temp_vars[var] = merged_expr
                
                if var in self.syms and var not in self.written_wires:
                    Expr("assign", merged_expr, self.syms[var].nxt())
                    self.written_wires.add(var)
            elif if_expr is not None:
                self.temp_vars[var] = if_expr
                
                if var in self.syms and var not in parent_scope and var not in self.written_wires:
                    Expr("assign", if_expr, self.syms[var].nxt())
                    self.written_wires.add(var)
    
    def visit_Assign(self, node):
        """Handle variable assignment"""
        if len(node.targets) != 1:
            raise ValueError("Only single assignment supported")
        
        target = node.targets[0]
        
        if isinstance(target, ast.Name):
            var_name = target.id
            result_expr = self._convert_expr(node.value)
            self.temp_vars[var_name] = result_expr
            
        elif isinstance(target, ast.Attribute) and target.attr in self.syms:
            wire_name = target.attr
            result_expr = self._convert_expr(node.value)
            self.temp_vars[wire_name] = result_expr
            
            if len(self.scopes) == 0:
                Expr("assign", result_expr, self.syms[wire_name].nxt())
                self.written_wires.add(wire_name)
    
    def visit_Return(self, node):
        """Assign interface wires from temp_vars (return values ignored)"""
        for item in self.env.intf:
            wire_name = item.name
            if wire_name in self.temp_vars and wire_name not in self.written_wires:
                result_expr = self.temp_vars[wire_name]
                Expr("assign", result_expr, self.syms[wire_name].nxt())
        
    def _convert_expr(self, expr):
        """Convert AST expression node to Expr object"""
        if isinstance(expr, ast.Call):
            return self._convert_call(expr)
        elif isinstance(expr, ast.BinOp):
            return self._convert_binop(expr)
        elif isinstance(expr, ast.Compare):
            return self._convert_compare(expr)
        elif isinstance(expr, ast.IfExp):
            return self._convert_ifexp(expr)
        elif isinstance(expr, ast.Attribute):
            return self._convert_attribute(expr)
        elif isinstance(expr, ast.Name):
            return self._convert_name(expr)
        elif isinstance(expr, ast.Constant):
            return self._convert_constant(expr)
        else:
            raise ValueError(f"Unsupported expression type: {type(expr).__name__}")
    
    def _convert_call(self, call):
        """Convert method/function call"""
        if isinstance(call.func, ast.Attribute):
            method = call.func.attr
            obj = call.func.value
            
            if method == 'argmax':
                obj_expr = self._convert_expr(obj)
                return Expr("argmax", obj_expr)
                
            elif method == 'item':
                return self._convert_expr(obj)
                
            elif isinstance(obj, ast.Name) and obj.id == 'self':
                return self._inline_method(method, call.args)
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
        """Convert min/max to Ite: min(a,b) -> Ite(Lt(a,b), a, b)"""
        if len(args) != 2:
            raise ValueError(f"{func_name}() requires 2 arguments")
        
        a_expr = self._convert_expr(args[0])
        b_expr = self._convert_expr(args[1])
        
        cmp_op = "cmp.lt" if func_name == 'min' else "cmp.gt"
        cmp_expr = Expr(cmp_op, a_expr, b_expr)
        
        return Expr("ite", cmp_expr, a_expr, b_expr)
    
    def _convert_binop(self, binop):
        """Convert binary operation"""
        left_expr = self._convert_expr(binop.left)
        
        if isinstance(binop.right, ast.Constant):
            right_expr = self._convert_constant(binop.right, left_expr.dtype())
        else:
            right_expr = self._convert_expr(binop.right)
        
        op_type = type(binop.op)
        if op_type not in self.BINARY_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        
        return Expr(self.BINARY_OPS[op_type], left_expr, right_expr)
    
    def _convert_compare(self, compare):
        """Convert comparison operation"""
        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            raise ValueError("Only simple comparisons supported")
        
        left_expr = self._convert_expr(compare.left)
        
        if isinstance(compare.comparators[0], ast.Constant):
            right_expr = self._convert_constant(compare.comparators[0], left_expr.dtype())
        else:
            right_expr = self._convert_expr(compare.comparators[0])
        
        op_type = type(compare.ops[0])
        if op_type not in self.COMPARE_OPS:
            raise ValueError(f"Unsupported comparison operator: {op_type.__name__}")
        
        return Expr(self.COMPARE_OPS[op_type], left_expr, right_expr)
    
    def _convert_ifexp(self, ifexp):
        """Convert ternary conditional to Ite"""
        cond_expr = self._convert_expr(ifexp.test)
        true_expr = self._convert_expr(ifexp.body)
        false_expr = self._convert_expr(ifexp.orelse)
        
        return Expr("ite", cond_expr, true_expr, false_expr)
    
    def _convert_name(self, name):
        """Convert variable reference"""
        var_name = name.id
        if var_name in self.temp_vars:
            return self.temp_vars[var_name]
        else:
            raise ValueError(f"Unknown variable: {var_name}")
    
    def _convert_constant(self, constant, target_dtype=None):
        """Convert constant literal to Expr
        
        Args:
            constant: AST Constant node
            target_dtype: Optional dtype to match. If None, defaults to float tensors.
        """
        value = constant.value
        
        if isinstance(value, bool):
            return Expr("const", value)
        elif isinstance(value, (int, float)):
            if target_dtype is None:
                # Default to float tensor
                tensor_data = torch.Tensor([float(value)])
            elif isinstance(target_dtype, DType.TensorInt):
                tensor_data = torch.tensor([int(value)], dtype=torch.long)
            elif isinstance(target_dtype, DType.TensorFloat):
                tensor_data = torch.tensor([float(value)], dtype=torch.float32)
            elif target_dtype == DType.Int:
                return Expr("const", int(value))
            elif target_dtype == DType.Float:
                return Expr("const", float(value))
            elif target_dtype == DType.Bool:
                return Expr("const", bool(value))
            else:
                # Fallback to float tensor
                tensor_data = torch.tensor([float(value)], dtype=torch.float32)
            return Expr("const", tensor_data)
        elif isinstance(value, (list, tuple)):
            tensor_data = torch.Tensor(value)
            return Expr("const", tensor_data)
        elif isinstance(value, torch.Tensor):
            return Expr("const", value)
        else:
            raise ValueError(f"Unsupported constant type: {type(value)}")
    
    def _convert_attribute(self, attr):
        """Convert attribute access (self.state, etc.)
        
        Checks temp_vars first, falls back to Sym for state reads.
        """
        if isinstance(attr.value, ast.Name) and attr.value.id == 'self':
            wire_name = attr.attr
            
            if wire_name in self.temp_vars:
                return self.temp_vars[wire_name]
            
            if wire_name in self.syms:
                return self.syms[wire_name]
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
