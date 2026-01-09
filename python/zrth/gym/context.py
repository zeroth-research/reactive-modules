from .backend import Wire, DType as MockDType
import zrth


def _get_real_dtype(dtype_str):
    """Convert string dtype to real DType enum"""
    dtype_map = {
        'Tensor': zrth.DType.C,
        'Bool': zrth.DType.D,
        'None': zrth.DType.C,
    }
    return dtype_map.get(dtype_str, zrth.DType.C)


class Context:
    """Wire registry for building reactive modules
    
    Manages wire ID generation and name-to-ID mapping.
    Follows the common-ctx API pattern for future compatibility.
    """
    
    def __init__(self):
        self.name_to_id = {}  # str -> int
        self._wire_counter = 0
        self._wire_dtypes = {}  # id -> dtype (for tracking)
        
    def fresh_var(self):
        """Create unnamed temporary variable
        
        Returns:
            Wire ID (int)
        """
        v = self._wire_counter
        self._wire_counter += 1
        self.name_to_id[f"__c_{v}"] = v
        return v
    
    def get_var(self, name: str):
        """Get or create named variable
        
        Args:
            name: Variable name
            
        Returns:
            Wire ID (int)
        """
        if name not in self.name_to_id:
            self.name_to_id[name] = self._wire_counter
            self._wire_counter += 1
        return self.name_to_id[name]
    
    def var(self, name: str, dtype):
        """Get or create named variable with dtype
        
        Args:
            name: Variable name
            dtype: Datatype (string like 'Tensor', 'Bool')
            
        Returns:
            Wire object
        """
        wire_id = self.get_var(name)
        self._wire_dtypes[wire_id] = dtype
        # Convert string dtype to RealDType enum
        real_dtype = _get_real_dtype(dtype)
        # NOTE: Wire constructor is Wire(dtype, id) NOT Wire(id, dtype)
        return Wire(real_dtype, wire_id)
    
    def tmp_wire(self, dtype):
        """Create temporary wire
        
        Args:
            dtype: Datatype (string like 'Tensor', 'Bool')
            
        Returns:
            Wire object
        """
        wire_id = self.fresh_var()
        self._wire_dtypes[wire_id] = dtype
        # Convert string dtype to RealDType enum
        real_dtype = _get_real_dtype(dtype)
        # NOTE: Wire constructor is Wire(dtype, id) NOT Wire(id, dtype)
        return Wire(real_dtype, wire_id)
    
    def has_wire(self, name):
        """Check if a wire is declared
        
        Args:
            name: Wire name
            
        Returns:
            True if wire exists
        """
        return name in self.name_to_id
    
    def num_wires(self):
        """Get number of created wires
        
        Returns:
            Number of wires
        """
        return self._wire_counter
    
    def get_name(self, wire_id: int):
        """Get name for a wire ID (if it has one)
        
        Args:
            wire_id: Wire ID
            
        Returns:
            Name or None
        """
        for name, wid in self.name_to_id.items():
            if wid == wire_id:
                return name
        return None
    
    def __str__(self):
        result = f"Context with {self._wire_counter} wires:\n"
        for name, wid in sorted(self.name_to_id.items(), key=lambda x: x[1]):
            dtype = self._wire_dtypes.get(wid, '?')
            result += f"  {name} (id={wid}): {dtype}\n"
        return result
