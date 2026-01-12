import torch
import inspect
import ast
import textwrap
from typing import Any
from .context import Context
from zrth import Wire, DType, IType, Term, Module, MyTensor

# Import Rust types (will fail until we have proper bindings)
# from zrth._zrth.torch import Wire, Term, Module, IType, DType
# When ready, uncomment above and remove mocks


def convert_to_module(ctx: Context, python_object: Any):
    """Convert a Python object to a reactive Module
    
    Args:
        ctx: Context for wire management
        python_object: Object to convert (QNetwork, SimpleEnv, etc.)
        
    Returns:
        Rust Module object
    """
    # Detect module type
    if hasattr(python_object, 'forward') and isinstance(python_object, torch.nn.Module):
        return _convert_torch_module(ctx, python_object)
    elif hasattr(python_object, 'step') and hasattr(python_object, 'reset'):
        return _convert_gym_env(ctx, python_object)
    else:
        raise ValueError(f"Don't know how to convert {type(python_object)}")


def _convert_torch_module(ctx: Context, module: torch.nn.Module):
    """Convert PyTorch module to reactive Module"""
    print(f"Converting PyTorch module: {type(module).__name__}")
    
    # Step 1: Trace the module
    # Get input size from first layer
    first_layer = None
    for layer in module.children():
        if isinstance(layer, torch.nn.Linear):
            first_layer = layer
            break
    
    if first_layer is None:
        raise ValueError("Can't find Linear layer to determine input shape")
    
    input_size = first_layer.in_features
    print(f"Detected input size: {input_size}")
    example_input = torch.zeros(1, input_size)
    
    traced = torch.jit.trace(module, example_input)
    print(f"Traced graph:\n{traced.graph}")
    
    # Step 2: Get output size
    last_layer = None
    for layer in module.children():
        if isinstance(layer, torch.nn.Linear):
            last_layer = layer
    
    output_size = last_layer.out_features if last_layer else 2
    print(f"Detected output size: {output_size}")
    
    # Step 3: Create wire pairs using Context
    # Use Tensor dtype for torch backend
    obs_latched = ctx.wire('observation_l', 'Tensor')
    obs_next = ctx.wire('observation_n', 'Tensor')
    obs_pair = (obs_latched, obs_next)
    
    qval_latched = ctx.wire('q_values_l', 'Tensor')
    qval_next = ctx.wire('q_values_n', 'Tensor')
    qval_pair = (qval_latched, qval_next)
    
    # Track named wire pairs for module metadata
    named_wire_pairs = {'observation': obs_pair, 'q_values': qval_pair}
    temp_wire_pairs = []
    
    # Step 4: Parse graph and generate Terms
    terms = _parse_torch_graph(traced.graph, module, ctx, temp_wire_pairs, obs_pair, qval_pair)
    
    print(f"Created wires: {list(named_wire_pairs.keys())}")
    print(f"Context now has: {ctx.num_wires()} wires")
    print(f"Generated {len(terms)} terms")
    
    # Step 5: Build Module
    # Terms are now in correct order (no need to reverse after fixing Term argument order!)
    all_wire_pairs = [obs_pair, qval_pair] + temp_wire_pairs
    
    print(f"Total wire pairs: {len(all_wire_pairs)}")
    
    module = Module.combinatorial(assign=terms, obs=all_wire_pairs)
    
    # Note: Can't attach metadata to Rust Module object
    # Store wire pairs mapping separately if needed
    
    return module


def _parse_torch_graph(graph, module, ctx, temp_wire_pairs, input_pair, output_pair):
    """Parse traced graph and generate Terms
    
    Args:
        graph: TorchScript graph
        module: Original PyTorch module (to get parameters)
        ctx: Context for creating temp wires
        temp_wire_pairs: List to collect temporary wire pairs
        input_pair: (latched, next) for input wire
        output_pair: (latched, next) for output wire
        
    Returns:
        List of Term representations
    """
    terms = []
    
    # Extract Linear layers to get weights and biases
    layers = []
    for layer in module.children():
        if isinstance(layer, torch.nn.Linear):
            layers.append(layer)
    
    if len(layers) == 0:
        raise ValueError("No Linear layers found in module")
    
    # Parse graph to detect activations between layers
    graph_str = str(graph)
    has_relu = 'aten::relu' in graph_str
    
    # Current wire pair tracks the output of the previous layer
    current_pair = input_pair
    
    for i, layer in enumerate(layers):
        is_last_layer = (i == len(layers) - 1)
        
        # MatMul: current -> temp_matmul
        # TODO: Add weight constants when torch backend supports them
        temp_matmul_l = ctx.tmp_wire('Tensor')
        temp_matmul_n = ctx.tmp_wire('Tensor')
        temp_matmul = (temp_matmul_l, temp_matmul_n)
        temp_wire_pairs.append(temp_matmul)
        terms.append(Term(IType.MatMul(), [temp_matmul[1]], [current_pair[1]]))
        
        # Add bias: temp_matmul -> temp_add
        # TODO: Add bias constants when torch backend supports them
        
        if is_last_layer:
            # Last layer outputs to the final output wire
            terms.append(Term(IType.Add(), [output_pair[1]], [temp_matmul[1]]))
        else:
            temp_add_l = ctx.tmp_wire('Tensor')
            temp_add_n = ctx.tmp_wire('Tensor')
            temp_add = (temp_add_l, temp_add_n)
            temp_wire_pairs.append(temp_add)
            terms.append(Term(IType.Add(), [temp_add[1]], [temp_matmul[1]]))
            
            # Apply ReLU if detected: Ite(Gt(x, 0), x, 0)
            if has_relu:
                # Gt(temp_add, 0)
                # TODO: Add zero constant when torch backend supports it
                gt_temp_l = ctx.tmp_wire('Bool')
                gt_temp_n = ctx.tmp_wire('Bool')
                gt_temp = (gt_temp_l, gt_temp_n)
                temp_wire_pairs.append(gt_temp)
                terms.append(Term(IType.Gt(), [gt_temp[1]], [temp_add[1]]))
                
                # Ite(gt_result, temp_add, 0) -> activated
                activated_l = ctx.tmp_wire('Tensor')
                activated_n = ctx.tmp_wire('Tensor')
                activated_pair = (activated_l, activated_n)
                temp_wire_pairs.append(activated_pair)
                terms.append(Term(IType.Ite(), [activated_pair[1]], [gt_temp[1], temp_add[1]]))
                
                current_pair = activated_pair
            else:
                current_pair = temp_add
    
    return terms


def _convert_gym_env(ctx: Context, env):
    """Convert gym environment to reactive Module"""
    print(f"Converting Gym environment: {type(env).__name__}")
    
    # Get wire declarations from the environment
    if not hasattr(env, 'extl') or not hasattr(env, 'intf') or not hasattr(env, 'prvt'):
        raise ValueError(f"Environment must declare extl, intf, prvt wires (SequentialModule)")
    
    # Parse the step() method to extract logic
    step_source = inspect.getsource(env.step)
    print(f"Parsing step() method...")
    
    # Create wire pairs for all declared wires using Context
    wire_pairs = {}
    
    # External wires (inputs)
    for name in env.extl:
        latched = ctx.wire(f'{name}_l', 'Tensor')
        next_wire = ctx.wire(f'{name}_n', 'Tensor')
        wire_pairs[name] = (latched, next_wire)
    
    # Interface wires (outputs)
    for name in env.intf:
        latched = ctx.wire(f'{name}_l', 'Tensor')
        next_wire = ctx.wire(f'{name}_n', 'Tensor')
        wire_pairs[name] = (latched, next_wire)
    
    # Private wires (internal state)
    for name in env.prvt:
        latched = ctx.wire(f'{name}_l', 'Tensor')
        next_wire = ctx.wire(f'{name}_n', 'Tensor')
        wire_pairs[name] = (latched, next_wire)
    
    # Track temp wire pairs (will be populated during parsing)
    temp_wire_pairs = []
    
    # Parse AST to extract computation
    terms = _parse_step_method(env, ctx, wire_pairs, temp_wire_pairs)
    
    print(f"Created wires: {wire_pairs.keys()}")
    print(f"Context now has: {ctx.num_wires()} wires")
    print(f"Generated {len(terms)} terms")
    
    # Build sequential module structure
    # zrth API: Module.sequential(init, update, obs=..., prvt=...)
    obs_list = [wire_pairs[name] for name in env.extl + env.intf]
    prvt_list = [wire_pairs[name] for name in env.prvt] + temp_wire_pairs
    
    # For sequential: init and update are the same terms for now
    # TODO: Separate init logic if needed
    module = Module.sequential(init=terms, update=terms, obs=obs_list, prvt=prvt_list)
    
    # Note: Can't attach metadata to Rust Module object
    # Store wire pairs mapping separately if needed
    
    return module


def _parse_step_method(env, ctx, wire_pairs, temp_wire_pairs):
    """Parse step() method and generate Terms automatically
    
    Uses AST analysis to extract computation logic and generate Terms.
    Supports: assignments, arithmetic, comparisons, conditionals, method calls
    
    Args:
        env: Environment object
        ctx: Context for creating temp wires
        wire_pairs: Named wire pairs dict
        temp_wire_pairs: List to collect temporary wire pairs
    """
    
    # Get source code and dedent to remove class indentation
    step_source = inspect.getsource(env.step)
    step_source = textwrap.dedent(step_source)
    tree = ast.parse(step_source)
    
    # Extract function body
    func_def = tree.body[0]
    if not isinstance(func_def, ast.FunctionDef):
        raise ValueError("Expected function definition")
    
    print(f"Analyzing {len(func_def.body)} statements in step() method...")
    
    # Create an AST visitor to convert statements to Terms
    visitor = StepMethodVisitor(env, ctx, wire_pairs, temp_wire_pairs)
    
    # Map function parameters to wire pairs
    # step(self, q_values) -> q_values is a parameter that maps to the q_values wire
    params = [arg.arg for arg in func_def.args.args if arg.arg != 'self']
    for param_name in params:
        if param_name in wire_pairs:
            visitor.temp_vars[param_name] = wire_pairs[param_name]
        else:
            print(f"Warning: parameter {param_name} not found in wire_pairs")
    
    for stmt in func_def.body:
        visitor.visit(stmt)
    
    return visitor.terms


class StepMethodVisitor(ast.NodeVisitor):
    """AST visitor to convert step() method to reactive Terms"""
    
    def __init__(self, env, ctx, wire_pairs, temp_wire_pairs):
        self.env = env
        self.ctx = ctx
        self.wire_pairs = wire_pairs
        self.temp_wire_pairs = temp_wire_pairs
        self.terms = []
        self.temp_vars = {}  # Maps variable names to wire pairs
        
    def visit_Assign(self, node):
        """Handle variable assignment: var = expr"""
        if len(node.targets) != 1:
            raise ValueError("Only single assignment supported")
        
        target = node.targets[0]
        if isinstance(target, ast.Name):
            var_name = target.id
            print(f"  Assignment: {var_name} = ...")
            
            # Generate wire pair and terms for the expression
            result_pair = self._convert_expr(node.value)
            
            # Check if this is updating a known wire (like self.state)
            if isinstance(target, ast.Attribute) and target.attr in self.wire_pairs:
                # Update the wire: connect result to the 'next' wire
                wire_name = target.attr
                self.wire_pairs[wire_name] = result_pair
            else:
                # Store as temporary variable
                self.temp_vars[var_name] = result_pair
        elif isinstance(target, ast.Attribute) and target.attr in self.wire_pairs:
            # Direct update to wire: self.state = expr
            wire_name = target.attr
            print(f"  Wire update: {wire_name} = ...")
            result_pair = self._convert_expr(node.value)
            
            # Connect result to the wire's 'next'
            self.terms.append(Term(IType.Id(), [self.wire_pairs[wire_name][1]], [result_pair[0]]))
        
    def _convert_expr(self, expr):
        """Convert an expression AST node to wire pair
        
        Returns: (latched, next) wire pair representing the expression result
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
            # Variable reference
            var_name = expr.id
            if var_name in self.temp_vars:
                return self.temp_vars[var_name]
            else:
                raise ValueError(f"Unknown variable: {var_name}")
        elif isinstance(expr, ast.Constant):
            # Constant value - wrap in Const term
            # TODO: Properly convert constant to tensor data
            const_wire = self.ctx.tmp_wire('Tensor')
            tensor_data = MyTensor([int(expr.value)])  # Convert to tensor format
            return (const_wire, None)  # Placeholder - needs proper Const term handling
        else:
            raise ValueError(f"Unsupported expression type: {type(expr).__name__}")
    
    def _convert_call(self, call):
        """Convert method call: obj.method(args)"""
        if isinstance(call.func, ast.Attribute):
            method = call.func.attr
            obj = call.func.value
            
            if method == 'argmax':
                # obj.argmax() -> Argmax term
                obj_pair = self._convert_expr(obj)
                result_l = self.ctx.tmp_wire('Tensor')
                result_n = self.ctx.tmp_wire('Tensor')
                result_pair = (result_l, result_n)
                self.temp_wire_pairs.append(result_pair)
                self.terms.append(Term(IType.Argmax(), [result_pair[1]], [obj_pair[0]]))
                return result_pair
            elif method == 'item':
                # .item() is just value extraction, pass through
                obj_pair = self._convert_expr(obj)
                return obj_pair
            elif isinstance(obj, ast.Name) and obj.id == 'self':
                # Method call on self - try to inline it
                return self._inline_method(method, call.args)
            else:
                raise ValueError(f"Unsupported method: {method}")
        elif isinstance(call.func, ast.Name):
            func_name = call.func.id
            
            if func_name == 'min':
                # min(a, b) -> Ite(Lt(a, b), a, b)
                if len(call.args) != 2:
                    raise ValueError("min() requires 2 arguments")
                a_pair = self._convert_expr(call.args[0])
                b_pair = self._convert_expr(call.args[1])
                
                # Lt(a, b)
                lt_l = self.ctx.tmp_wire('Tensor')
                lt_n = self.ctx.tmp_wire('Tensor')
                lt_pair = (lt_l, lt_n)
                self.temp_wire_pairs.append(lt_pair)
                self.terms.append(Term(IType.Lt(), [lt_pair[1]], [a_pair[0], b_pair[0]]))
                
                # Ite(lt, a, b)
                result_l = self.ctx.tmp_wire('Tensor')
                result_n = self.ctx.tmp_wire('Tensor')
                result_pair = (result_l, result_n)
                self.temp_wire_pairs.append(result_pair)
                self.terms.append(Term(IType.Ite(), [result_pair[1]], [lt_pair[0], a_pair[0], b_pair[0]]))
                return result_pair
                
            elif func_name == 'max':
                # max(a, b) -> Ite(Gt(a, b), a, b)
                if len(call.args) != 2:
                    raise ValueError("max() requires 2 arguments")
                a_pair = self._convert_expr(call.args[0])
                b_pair = self._convert_expr(call.args[1])
                
                # Gt(a, b)
                gt_l = self.ctx.tmp_wire('Tensor')
                gt_n = self.ctx.tmp_wire('Tensor')
                gt_pair = (gt_l, gt_n)
                self.temp_wire_pairs.append(gt_pair)
                self.terms.append(Term(IType.Gt(), [gt_pair[1]], [a_pair[0], b_pair[0]]))
                
                # Ite(gt, a, b)
                result_l = self.ctx.tmp_wire('Tensor')
                result_n = self.ctx.tmp_wire('Tensor')
                result_pair = (result_l, result_n)
                self.temp_wire_pairs.append(result_pair)
                self.terms.append(Term(IType.Ite(), [result_pair[1]], [gt_pair[0], a_pair[0], b_pair[0]]))
                return result_pair
            else:
                raise ValueError(f"Unsupported function: {func_name}")
        else:
            raise ValueError(f"Unsupported call type: {type(call.func).__name__}")
    
    def _convert_binop(self, binop):
        """Convert binary operation: a + b, a - b, etc."""
        left_pair = self._convert_expr(binop.left)
        right_pair = self._convert_expr(binop.right)
        
        op_map = {
            ast.Add: 'Add',
            ast.Sub: 'Sub',
            ast.Mult: 'Mul',
            ast.Div: 'Div',
        }
        
        op_type = type(binop.op)
        if op_type not in op_map:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        
        itype = op_map[op_type]
        result_l = self.ctx.tmp_wire('Tensor')
        result_n = self.ctx.tmp_wire('Tensor')
        result_pair = (result_l, result_n)
        self.temp_wire_pairs.append(result_pair)
        # Map string to IType enum variant
        itype_enum = getattr(IType, itype)()
        self.terms.append(Term(itype_enum, [result_pair[1]], [left_pair[0], right_pair[0]]))
        return result_pair
    
    def _convert_compare(self, compare):
        """Convert comparison: a == b, a < b, etc."""
        if len(compare.ops) != 1 or len(compare.comparators) != 1:
            raise ValueError("Only simple comparisons supported")
        
        left_pair = self._convert_expr(compare.left)
        right_pair = self._convert_expr(compare.comparators[0])
        
        op_map = {
            ast.Eq: 'Eq',
            ast.NotEq: 'Neq',
            ast.Lt: 'Lt',
            ast.LtE: 'Le',
            ast.Gt: 'Gt',
            ast.GtE: 'Ge',
        }
        
        op_type = type(compare.ops[0])
        if op_type not in op_map:
            raise ValueError(f"Unsupported comparison operator: {op_type.__name__}")
        
        itype = op_map[op_type]
        result_l = self.ctx.tmp_wire('Tensor')
        result_n = self.ctx.tmp_wire('Tensor')
        result_pair = (result_l, result_n)
        self.temp_wire_pairs.append(result_pair)
        # Map string to IType enum variant
        itype_enum = getattr(IType, itype)()
        self.terms.append(Term(itype_enum, [result_pair[1]], [left_pair[0], right_pair[0]]))
        return result_pair
    
    def _convert_ifexp(self, ifexp):
        """Convert ternary if: a if cond else b -> Ite(cond, a, b)"""
        cond_pair = self._convert_expr(ifexp.test)
        true_pair = self._convert_expr(ifexp.body)
        false_pair = self._convert_expr(ifexp.orelse)
        
        result_l = self.ctx.tmp_wire('Tensor')
        result_n = self.ctx.tmp_wire('Tensor')
        result_pair = (result_l, result_n)
        self.temp_wire_pairs.append(result_pair)
        self.terms.append(Term(IType.Ite(), [result_pair[1]], [cond_pair[0], true_pair[0], false_pair[0]]))
        return result_pair
    
    def _convert_attribute(self, attr):
        """Convert attribute access: self.state, self.observation, etc."""
        if isinstance(attr.value, ast.Name) and attr.value.id == 'self':
            wire_name = attr.attr
            if wire_name in self.wire_pairs:
                return self.wire_pairs[wire_name]
            else:
                raise ValueError(f"Unknown wire: {wire_name}")
        else:
            raise ValueError(f"Unsupported attribute access: {ast.unparse(attr)}")
    
    def _inline_method(self, method_name, args):
        """Inline a simple method call by extracting its return expression
        
        Only supports simple methods that return a single expression.
        """
        if not hasattr(self.env, method_name):
            raise ValueError(f"Method not found: {method_name}")
        
        method = getattr(self.env, method_name)
        method_source = inspect.getsource(method)
        method_source = textwrap.dedent(method_source)
        method_tree = ast.parse(method_source)
        
        method_def = method_tree.body[0]
        if not isinstance(method_def, ast.FunctionDef):
            raise ValueError(f"Expected function definition for {method_name}")
        
        # Find the return statement
        return_stmt = None
        for stmt in method_def.body:
            if isinstance(stmt, ast.Return):
                return_stmt = stmt
                break
        
        if return_stmt is None or return_stmt.value is None:
            raise ValueError(f"Method {method_name} has no return value")
        
        # Convert the return expression
        return self._convert_expr(return_stmt.value)