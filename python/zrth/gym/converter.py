import torch
import inspect
import ast
import textwrap
from typing import Any
from ..context import Context
from zrth import Wire, DType, IType, Term, Module
from zrth.expr import matmul_dtype


def convert_to_module(ctx: Context, python_object: Any):
    """Convert a Python object to a reactive Module

    Args:
        ctx: Context for wire management
        python_object: Object to convert (QNetwork, SimpleEnv, etc.)

    Returns:
        Rust Module object
    """
    # Detect module type
    if hasattr(python_object, "forward") and isinstance(python_object, torch.nn.Module):
        return _convert_torch_module(ctx, python_object)
    elif hasattr(python_object, "step") and hasattr(python_object, "reset"):
        return _convert_gym_env(ctx, python_object)
    else:
        raise ValueError(f"Don't know how to convert {type(python_object)}")


def _convert_torch_module(ctx: Context, module: torch.nn.Module):
    # Trace the module
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

    # Validate single input/output constraint
    if len(module.extl) != 1 or len(module.intf) != 1:
        raise ValueError(
            f"Only single input/output modules supported. Got {len(module.extl)} inputs, {len(module.intf)} outputs"
        )

    # Create wire pairs dynamically from module metadata
    wire_pairs = {}
    for name in module.extl + module.intf:
        latched = ctx.wire(f"{name}_l", DType.Tensor(example_input.size()))
        next_wire = ctx.wire(f"{name}_n", DType.Tensor(example_input.size()))
        wire_pairs[name] = (latched, next_wire)

    # Extract input and output pairs
    input_pair = wire_pairs[module.extl[0]]
    output_pair = wire_pairs[module.intf[0]]

    # Parse TorchScript graph to extract operation sequence
    operations = _parse_torchscript_graph(traced.graph, module)

    # Translate operations to Terms
    terms = []
    current_wire = input_pair[1]  # Start with input's 'next' wire

    for op in operations:
        if op["type"] == "linear":
            new_terms, current_wire = _translate_linear(ctx, current_wire, op["layer"])
            terms.extend(new_terms)
        elif op["type"] == "relu":
            new_terms, current_wire = _translate_relu(ctx, current_wire)
            terms.extend(new_terms)
        else:
            raise ValueError(f"Unsupported operation: {op['type']}")

    # Connect final wire to output
    terms.append(Term(IType.Id(), [output_pair[1]], [current_wire]))

    # Build Module
    obs = [wire_pairs[name] for name in module.extl + module.intf]

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

    # Maps: layer names to layer objects
    layer_map = {}
    for name, layer in module.named_children():
        layer_map[name] = layer

    # Maps: output_name -> (layer_name, layer_object)
    getattr_outputs = {}

    # Parse graph nodes in execution order
    for node in graph.nodes():
        kind = node.kind()

        if kind == "prim::GetAttr":
            # GetAttr extracts a layer: %fc1 = prim::GetAttr[name="fc1"](%self)
            layer_name = node.s("name")  # Get the 'name' attribute
            output_name = node.output().debugName()  # e.g., "%fc1"

            if layer_name in layer_map:
                getattr_outputs[output_name] = (layer_name, layer_map[layer_name])

        elif kind == "prim::CallMethod":
            # CallMethod calls layer.forward(): %result = prim::CallMethod[name="forward"](%fc1, %input)
            method_name = node.s("name")

            if method_name == "forward":
                # First input is the layer object, check if it's a tracked layer
                inputs = list(node.inputs())
                if len(inputs) >= 1:
                    layer_ref = inputs[0].debugName()

                    if layer_ref in getattr_outputs:
                        layer_name, layer_obj = getattr_outputs[layer_ref]

                        # Check layer type and add appropriate operation
                        if isinstance(layer_obj, torch.nn.Linear):
                            operations.append({"type": "linear", "layer": layer_obj})
                        else:
                            raise ValueError(
                                f"Unsupported layer type: {type(layer_obj).__name__}"
                            )

        elif kind == "aten::relu":
            operations.append({"type": "relu"})

        # Ignore other internal operations
        elif kind in ["prim::Constant", "prim::ListConstruct"]:
            continue

        # Everything else is unsupported
        else:
            raise ValueError(f"Unsupported operation: {kind}")

    return operations


def _translate_linear(ctx, input_wire, layer):
    """Translate Linear layer to Terms

    Args:
        ctx: Context for creating temp wires
        input_wire: Input wire ID
        layer: torch.nn.Linear layer object

    Returns:
        (terms, output_wire): List of Terms and output wire ID
    """
    terms = []

    # Create weight constant
    weight_tensor = layer.weight.data
    weight_wire = ctx.tmp_wire(DType.Tensor(weight_tensor.size()))
    terms.append(Term(IType.Tensor(weight_tensor), [weight_wire]))

    # MatMul: input × weight -> temp_matmul
    temp_matmul = ctx.tmp_wire(matmul_dtype(weight_wire.dtype(), input_wire.dtype()))
    terms.append(Term(IType.MatMul(), [temp_matmul], [input_wire, weight_wire]))

    # Create bias constant
    bias_tensor = layer.bias.data
    bias_wire = ctx.tmp_wire(DType.Tensor(bias_tensor.size()))
    terms.append(Term(IType.Tensor(bias_tensor), [bias_wire]))

    # Add: matmul + bias -> output
    output_wire = ctx.tmp_wire(bias_wire.dtype())
    terms.append(Term(IType.Add(), [output_wire], [temp_matmul, bias_wire]))

    return terms, output_wire


def _translate_relu(ctx, input_wire):
    """Translate ReLU activation to Terms

    Implements: max(0, x) = Ite(Gt(x, 0), x, 0)

    Args:
        ctx: Context for creating temp wires
        input_wire: Input wire ID

    Returns:
        (terms, output_wire): List of Terms and output wire ID
    """
    terms = []

    # Create zero constant
    zero_tensor = torch.Tensor([[0.0], [1.0]])
    zero_wire = ctx.tmp_wire(DType.Tensor(zero_tensor.size()))
    terms.append(Term(IType.Tensor(zero_tensor), [zero_wire]))

    # Gt(input, 0)
    gt_wire = ctx.tmp_wire(DType.Bool())
    terms.append(Term(IType.Gt(), [gt_wire], [input_wire, zero_wire]))

    # Ite(gt_result, input, 0) -> activated
    output_wire = ctx.tmp_wire(DType.Tensor(zero_tensor.size()))
    terms.append(Term(IType.Ite(), [output_wire], [gt_wire, input_wire, zero_wire]))

    return terms, output_wire


def _convert_gym_env(ctx: Context, env):
    return NotImplementedError("Gym environment conversion not implemented yet")
