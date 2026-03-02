from .zrth import (
    Wire,
    DType,
    IType,
    Term,
    Transition,
    Module,
)


from typing import Generator


#####################################################################
# IType and DType
#####################################################################


# Add type aliases and convenient function
def mk_DTypeBool(shape: None | list[int] = None) -> DType:
    """
    Create a Bool DType. If shape is given, create a tensor of bools
    """
    if shape is None:
        shape = [1]
    return DType.TensorBool(shape)


def mk_DTypeInt(shape: None | list[int] = None) -> DType:
    """
    Create a Int DType. If shape is given, create a tensor of bools
    """
    if shape is None:
        shape = [1]
    return DType.TensorInt(shape)


def mk_DTypeFloat(shape: None | list[int] = None) -> DType:
    """
    Create a Float DType. If shape is given, create a tensor of bools
    """
    if shape is None:
        shape = [1]
    return DType.TensorFloat(shape)


def mk_DTypeReal(shape: None | list[int] = None) -> DType:
    """
    Create a Real DType. If shape is given, create a tensor of bools
    """
    if shape is None:
        shape = [1]
    return DType.TensorReal(shape)


#####################################################################
# Wire/Term helpers
#####################################################################

DType.Bool = mk_DTypeBool  # ty: ignore
DType.Int = mk_DTypeInt  # ty: ignore
DType.Float = mk_DTypeFloat  # ty: ignore
DType.Real = mk_DTypeReal  # ty: ignore


# Add type aliases to the DType object
def Bool(*shape):
    return DType.Bool([*shape])


def Int(*shape):
    return DType.Int([*shape])


def Real(*shape):
    return DType.Real([*shape])


def Float(*args):
    return DType.Float([*args])


__all__ = [
    "Wire",
    "DType",
    "IType",
    "Term",
    "Module",
    "Transition",
]
