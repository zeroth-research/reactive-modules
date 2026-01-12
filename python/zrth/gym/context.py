import zrth
from zrth import Wire

_DTYPE_TENSOR = zrth.DType.Tensor()
_DTYPE_BOOL = zrth.DType.Bool()
_DTYPE_INT = zrth.DType.Int()
_DTYPE_FLOAT = zrth.DType.Float()

_DTYPE_MAP = {
    'Tensor': _DTYPE_TENSOR,
    'Bool': _DTYPE_BOOL,
    'Int': _DTYPE_INT,
    'Float': _DTYPE_FLOAT,
}


def _get_real_dtype(dtype_str):
    """Convert string dtype to real DType enum"""
    return _DTYPE_MAP.get(dtype_str, _DTYPE_TENSOR)


class Context:
    """Wire registry for building reactive modules
    
    Manages wire ID generation and name-to-ID mapping.
    Follows the common-ctx API pattern for future compatibility.
    """
    
    def __init__(self):
        self.name_to_id = {}  # str -> int
        self.id_to_name = {}  # int -> str
        self._wire_counter = 0
        self._wire_dtypes = {}  # id -> DType enum
        
    def fresh_wire_id(self):
        """Create unnamed temporary wire ID
        
        Returns:
            Wire ID (int)
        """
        v = self._wire_counter
        self._wire_counter += 1
        name = f"__c_{v}"
        self.name_to_id[name] = v
        self.id_to_name[v] = name
        return v
    
    def get_wire_id(self, name: str):
        """Get or create named wire ID
        
        Args:
            name: Wire name
            
        Returns:
            Wire ID (int)
        """
        if name not in self.name_to_id:
            wid = self._wire_counter
            self._wire_counter += 1
            self.name_to_id[name] = wid
            self.id_to_name[wid] = name
        return self.name_to_id[name]
    
    def wire(self, name: str, dtype):
        """Get or create named wire with dtype
        
        Args:
            name: Wire name
            dtype: Datatype (string like 'Tensor', 'Bool')
            
        Returns:
            Wire object
        """
        wire_id = self.get_wire_id(name)
        real_dtype = _get_real_dtype(dtype)
        self._wire_dtypes[wire_id] = real_dtype
        return Wire(real_dtype, wire_id)
    
    def tmp_wire(self, dtype):
        """Create temporary wire
        
        Args:
            dtype: Datatype (string like 'Tensor', 'Bool')
            
        Returns:
            Wire object
        """
        wire_id = self.fresh_wire_id()
        real_dtype = _get_real_dtype(dtype)
        self._wire_dtypes[wire_id] = real_dtype
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
        return self.id_to_name.get(wire_id)
    
    def __str__(self):
        result = f"Context with {self._wire_counter} wires:\n"
        for name, wid in sorted(self.name_to_id.items(), key=lambda x: x[1]):
            dtype = self._wire_dtypes.get(wid, _DTYPE_TENSOR)
            result += f"  {name} (id={wid}): {dtype}\n"
        return result
