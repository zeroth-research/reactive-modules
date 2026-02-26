from .zrth import (
    Wire,
    DType,
    IType,
    Term,
    Module,
)



#####################################################################
# IType and DType
#####################################################################


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
]
