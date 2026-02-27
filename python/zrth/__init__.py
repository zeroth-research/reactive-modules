from .zrth import (
    Wire,
    DType,
    IType,
    Term,
    Module,
)


from .smv import parse_smv


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


def UWord(width: int):
    return DType.UWord(width)


def SWord(width: int):
    return DType.SWord(width)



__all__ = [
    "Wire",
    "DType",
    "IType",
    "Term",
    "Module",
]
