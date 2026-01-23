import torch
import inspect
import ast
import textwrap
from typing import Any
from zrth.context import get_ctx
from zrth import Wire, DType, IType, Term, Module
from zrth.expr import matmul_dtype


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
            new_terms, current_wire = _translate_linear(current_wire, op["layer"])
            terms.extend(new_terms)
        elif op["type"] == "relu":
            new_terms, current_wire = _translate_relu(current_wire)
            terms.extend(new_terms)
        else:
            raise ValueError(f"Unsupported operation: {op['type']}")

    terms.append(Term(IType.Id(), [module.intf[0].nxt().wire()], [current_wire]))

    obs = [(s.wire(), s.nxt().wire()) for s in module.extl + module.intf]

    # TODO: Return terms and obs instead of creating the module here.
    # The issue is that Module.combinatorial expects different args then Module.sequential.
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


def _translate_linear(input_wire, layer):
    """Translate Linear layer to Terms

    Args:
        input_wire: Input wire ID
        layer: torch.nn.Linear layer object

    Returns:
        (terms, output_wire): List of Terms and output wire ID
    """
    ctx = get_ctx()
    terms = []

    weight_tensor = layer.weight.data
    weight_wire = ctx.tmp_wire(DType.TensorFloat(weight_tensor.size()))
    terms.append(Term(IType.Tensor(weight_tensor), [weight_wire]))

    temp_matmul = ctx.tmp_wire(matmul_dtype(weight_wire.dtype(), input_wire.dtype()))
    terms.append(Term(IType.MatMul(), [temp_matmul], [input_wire, weight_wire]))

    bias_tensor = layer.bias.data
    bias_wire = ctx.tmp_wire(DType.TensorFloat(bias_tensor.size()))
    terms.append(Term(IType.Tensor(bias_tensor), [bias_wire]))

    output_wire = ctx.tmp_wire(bias_wire.dtype())
    terms.append(Term(IType.Add(), [output_wire], [temp_matmul, bias_wire]))

    return terms, output_wire


def _translate_relu(input_wire):
    """Translate ReLU activation to Terms

    Implements: max(0, x) = Ite(Gt(x, 0), x, 0)

    Args:
        input_wire: Input wire ID

    Returns:
        (terms, output_wire): List of Terms and output wire ID
    """
    ctx = get_ctx()
    terms = []

    zero_tensor = torch.Tensor([[0.0], [1.0]])
    zero_wire = ctx.tmp_wire(DType.TensorFloat(zero_tensor.size()))
    terms.append(Term(IType.TensorFloat(zero_tensor), [zero_wire]))

    gt_wire = ctx.tmp_wire(DType.Bool)
    terms.append(Term(IType.Gt(), [gt_wire], [input_wire, zero_wire]))

    output_wire = ctx.tmp_wire(DType.TensorFloat(zero_tensor.size()))
    terms.append(Term(IType.Ite(), [output_wire], [gt_wire, input_wire, zero_wire]))

    return terms, output_wire


# ============================================================================
# Gym Environment Conversion
# ============================================================================

def _convert_gym_env(env):
    """Convert gym environment to reactive Module
    
    Args:
        env: Environment object with reset() and step() methods
        
    Returns:
        Rust Module object
    """
    # Validate wire declarations
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
    
    required_intf_wires = [wire_pairs[item.name][1] for item in env.intf]
    _ensure_wires_assigned(init_terms, required_intf_wires, mode='init')
    _ensure_wires_assigned(update_terms, required_intf_wires, mode='update')
    
    return Module.sequential(init=init_terms, update=update_terms, obs=obs, prvt=prvt)


def _parse_method(env, method, wire_pairs):
    """Parse Python method and generate Terms using AST analysis"""
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
    
    return visitor.terms


def _ensure_wires_assigned(terms, required_wires, mode):
    """Validate all required wires are assigned, raising clear errors for any missing"""
    written_ids = set()
    for term in terms:
        prefix = str(term).split(';')[0]
        for part in prefix.split():
            if part.startswith('w') and part[1:].split(',')[0].isdigit():
                written_ids.add(part.rstrip(';,'))
    
    for wire in required_wires:
        wire_id = str(wire).split(' :')[0].strip()
        if f'w{wire_id}' not in written_ids:
            raise ValueError(
                f"Interface wire (id={wire_id}) was not assigned in {mode}() method. "
                f"Ensure you define a variable matching your intf declaration. "
                f"Variable names must match exactly with the interface wire names."
            )


class MethodVisitor(ast.NodeVisitor):
    """AST visitor to convert Python methods to reactive Terms
    
    Uses single wire IDs for temp computation, wire pairs only for state.
    """
    
    BINARY_OPS = {
        ast.Add: 'Add',
        ast.Sub: 'Sub',
        ast.Mult: 'Mul',
        ast.Div: 'Div',
    }
    
    COMPARE_OPS = {
        ast.Eq: 'Eq',
        ast.NotEq: 'Neq',
        ast.Lt: 'Lt',
        ast.LtE: 'Le',
        ast.Gt: 'Gt',
        ast.GtE: 'Ge',
    }
    
    def __init__(self, env, wire_pairs):
        self.env = env
        self.ctx = get_ctx()
        self.wire_pairs = wire_pairs
        self.terms = []
        self.temp_vars = {}
        self.scopes = []
        self.written_wires = set()
    
    def visit_If(self, node):
        """Handle if/else statement with SSA (Static Single Assignment)
        
        SSA ensures each wire is written exactly once by:
        1. Evaluating both branches in separate scopes
        2. Merging diverged variables with Ite terms at the merge point
        3. Writing the merged result to output wires
        
        This eliminates the "write after write" error and provides
        semantically correct branch merging.
        
        Example:
            if action == 1:
                self.state = self.state + 1
            else:
                self.state = self.state - 1
        
        Generates:
            - Both branches compute their values (no output wire writes)
            - Ite merges the two computed values
            - Single write of merged value to self.state output wire
        """
        cond_wire = self._convert_expr(node.test)
        
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
            if_wire = if_scope_after.get(var, parent_scope.get(var))
            else_wire = else_scope_after.get(var, parent_scope.get(var))
            
            if if_wire != else_wire:
                merged_wire = self.ctx.tmp_wire(if_wire.dtype())
                self.terms.append(Term(IType.Ite(), [merged_wire], [cond_wire, if_wire, else_wire]))
                self.temp_vars[var] = merged_wire
                
                # If this is a state wire, write the merged value to output
                if var in self.wire_pairs and var not in self.written_wires:
                    output_wire = self.wire_pairs[var][1]
                    term = Term(IType.Id(), [output_wire], [merged_wire])
                    self.terms.append(term)
                    self.written_wires.add(var)
            elif if_wire is not None:
                self.temp_vars[var] = if_wire
                
                # Only write if this is a new variable in this scope
                if var in self.wire_pairs and var not in parent_scope and var not in self.written_wires:
                    output_wire = self.wire_pairs[var][1]
                    term = Term(IType.Id(), [output_wire], [if_wire])
                    self.terms.append(term)
                    self.written_wires.add(var)
    
    def visit_Assign(self, node):
        """Handle variable assignment: var = expr"""
        if len(node.targets) != 1:
            raise ValueError("Only single assignment supported")
        
        target = node.targets[0]
        
        if isinstance(target, ast.Name):
            var_name = target.id
            result_wire = self._convert_expr(node.value)
            self.temp_vars[var_name] = result_wire
            
        elif isinstance(target, ast.Attribute) and target.attr in self.wire_pairs:
            wire_name = target.attr
            result_wire = self._convert_expr(node.value)
            self.temp_vars[wire_name] = result_wire
            
            if len(self.scopes) == 0:
                # Not in conditional - write directly to output wire
                output_wire = self.wire_pairs[wire_name][1]
                term = Term(IType.Id(), [output_wire], [result_wire])
                self.terms.append(term)
                self.written_wires.add(wire_name)
    
    def visit_Return(self, node):
        """Map interface wires by name lookup in temp_vars (return values ignored)"""
        for item in self.env.intf:
            wire_name = item.name
            if wire_name in self.temp_vars and wire_name not in self.written_wires:
                result_wire = self.temp_vars[wire_name]
                output_wire = self.wire_pairs[wire_name][1]
                term = Term(IType.Id(), [output_wire], [result_wire])
                self.terms.append(term)
        
    def _convert_expr(self, expr):
        """Convert an expression AST node to single wire ID
        
        Returns: wire ID (int) representing the expression result
        """
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
        """Convert method call: obj.method(args)"""
        if isinstance(call.func, ast.Attribute):
            method = call.func.attr
            obj = call.func.value
            
            if method == 'argmax':
                obj_wire = self._convert_expr(obj)
                result = self.ctx.tmp_wire(DType.TensorFloat([1]))
                self.terms.append(Term(IType.Argmax(), [result], [obj_wire]))
                return result
                
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
        """Convert min/max to conditional: min(a,b) -> Ite(Lt(a,b), a, b)"""
        if len(args) != 2:
            raise ValueError(f"{func_name}() requires 2 arguments")
        
        a_wire = self._convert_expr(args[0])
        b_wire = self._convert_expr(args[1])
        
        cmp_type = IType.Lt() if func_name == 'min' else IType.Gt()
        cmp_wire = self.ctx.tmp_wire(DType.Bool())
        self.terms.append(Term(cmp_type, [cmp_wire], [a_wire, b_wire]))
        
        result = self.ctx.tmp_wire(a_wire.dtype())
        self.terms.append(Term(IType.Ite(), [result], [cmp_wire, a_wire, b_wire]))
        return result
    
    def _convert_binop(self, binop):
        """Convert binary operation: a + b, a - b, etc."""
        left_wire = self._convert_expr(binop.left)
        right_wire = self._convert_expr(binop.right)
        
        op_type = type(binop.op)
        if op_type not in self.BINARY_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        
        result = self.ctx.tmp_wire(left_wire.dtype())
        itype_enum = getattr(IType, self.BINARY_OPS[op_type])()
        self.terms.append(Term(itype_enum, [result], [left_wire, right_wire]))
        return result
    
    def _convert_compare(self, compare):
        """Convert comparison: a == b, a < b, etc."""
        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            raise ValueError("Only simple comparisons supported")
        
        left_wire = self._convert_expr(compare.left)
        right_wire = self._convert_expr(compare.comparators[0])
        
        op_type = type(compare.ops[0])
        if op_type not in self.COMPARE_OPS:
            raise ValueError(f"Unsupported comparison operator: {op_type.__name__}")
        
        result = self.ctx.tmp_wire(DType.Bool())
        itype_enum = getattr(IType, self.COMPARE_OPS[op_type])()
        self.terms.append(Term(itype_enum, [result], [left_wire, right_wire]))
        return result
    
    def _convert_ifexp(self, ifexp):
        """Convert ternary if: a if cond else b -> Ite(cond, a, b)"""
        cond_wire = self._convert_expr(ifexp.test)
        true_wire = self._convert_expr(ifexp.body)
        false_wire = self._convert_expr(ifexp.orelse)
        
        result = self.ctx.tmp_wire(true_wire.dtype())
        self.terms.append(Term(IType.Ite(), [result], [cond_wire, true_wire, false_wire]))
        return result
    
    def _convert_name(self, name):
        """Convert variable reference"""
        var_name = name.id
        if var_name in self.temp_vars:
            return self.temp_vars[var_name]
        else:
            raise ValueError(f"Unknown variable: {var_name}")
    
    def _convert_constant(self, constant):
        """Convert constant literals to tensor wires"""
        value = constant.value
        
        if isinstance(value, (int, float, list, tuple, torch.Tensor)):
            if isinstance(value, (int, float)):
                tensor_data = torch.Tensor([float(value)])
            else:
                tensor_data = torch.Tensor(value)
            
            const_wire = self.ctx.tmp_wire(DType.TensorFloat(tensor_data.size()))
            self.terms.append(Term(IType.Tensor(tensor_data), [const_wire]))
            return const_wire
        else:
            raise ValueError(f"Unsupported constant type: {type(value)}")
    
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
        """Inline a simple method call by extracting its return expression"""
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
