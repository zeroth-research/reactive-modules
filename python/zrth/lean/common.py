"""Re-exports what `zrth.lean.common` held before it was split.

New code imports from the modules themselves: `errors` (`Refused`),
`sorts`, `layout`, `literals` and `context` (`LeanContext`). This stays
for the importers not migrated: `magic/learn.py`, which moves on
another branch."""

# ruff: noqa: F401
from .errors import (
    Refused,
)
from .sorts import (
    dtype_shape,
    _is_scalar_shape,
    lean_elem_type,
    dtype_to_lean_type,
    itype_name,
    _CONST_VARIANTS,
    is_constant_name,
    _is_scalar_tensor,
    _is_scalar_wire,
    _flat_element_type,
)
from .layout import (
    _accessor,
    _flat_indices,
    _flat_size,
    Slot,
    FlatLayout,
    flat_layout,
    _mat_from_scalars,
)
from .literals import (
    ConstantRegistry,
    _constant_expr,
    _get_dtype_item,
    _tensor_to_lean_def,
    _float_literal,
    _tensor_to_lean_inline,
    _tensor_to_lean_scalar,
    linear_list_literals,
)
from .context import (
    LeanContext,
    _any_float_wire,
    _bind_wires,
)
