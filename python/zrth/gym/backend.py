"""Backend interface for reactive modules - uses real Rust bindings with mock DType/IType"""

import torch
from zrth import Wire, Term, Module
from zrth import DType as RealDType, IType as RealIType


# ============================================================================
# Mock DType/IType (temporary until torch backend implementations exist)
# These are used for naming and will be mapped to toy backend types
# ============================================================================

class MockDType:
    """Mock DType enum - maps to toy backend for now"""
    Tensor = "Tensor"  # Will map to RealDType.C
    Bool = "Bool"      # Will map to RealDType.D
    None_ = "None"     # Will map to RealDType.C


class MockIType:
    """Mock IType with variants - maps to toy backend for now"""
    # All operations map to toy backend IType.A for now
    # Arithmetic
    Add = "Add"
    Sub = "Sub"
    Mul = "Mul"
    Div = "Div"
    MatMul = "MatMul"
    
    # Comparison
    Eq = "Eq"
    Neq = "Neq"
    Lt = "Lt"
    Le = "Le"
    Gt = "Gt"
    Ge = "Ge"
    
    # Logical
    And = "And"
    Or = "Or"
    Not = "Not"
    
    # Control flow
    Ite = "Ite"
    Choose = "Choose"
    Filter = "Filter"
    
    # Aggregation
    Sum = "Sum"
    Prod = "Prod"
    
    # Special
    Id = "Id"
    Argmax = "Argmax"
    
    @staticmethod
    def Const(tensor):
        """Create a constant term"""
        return ("Const", tensor)


# Use mock DType/IType for naming (will be mapped to real types)
DType = MockDType
IType = MockIType


# ============================================================================
# Helper functions
# ============================================================================

def create_wire(dtype_str, wire_id):
    """Create a Wire with string dtype (converts to proper DType)
    
    Maps our torch-style types to toy backend types for now.
    """
    # Map our types to toy backend
    dtype_map = {
        'Tensor': RealDType.C,
        'Bool': RealDType.D,
        'None': RealDType.C,
    }
    
    real_dtype = dtype_map.get(dtype_str, RealDType.C)
    return Wire(real_dtype, wire_id)

def create_const(value):
    """Create a constant IType
    
    Maps to toy backend IType.C with tensor parameter.
    """
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value)
    # Use toy backend IType.C with tensor parameter
    # MyTensor expects integers, so convert
    from mylib import MyTensor
    mytensor = MyTensor([int(x) for x in value.flatten().tolist()])
    return RealIType.C(mytensor)


def create_term(itype_str, inputs, outputs):
    """Create a Term from string itype name
    
    Args:
        itype_str: String name like "Add", "MatMul", etc.
        inputs: List of Wire or Const objects (will be READ by the term)
        outputs: List of Wire objects (will be WRITTEN by the term)
        
    Maps all operations to toy backend IType.A for now.
    """
    # For now, map everything to IType.A (toy backend generic operation)
    # When torch backend is ready, this will use actual ITypes
    itype = RealIType.A()
    
    return Term(itype, outputs, inputs)
