from .converter import convert_to_module


class Module:
    """Base class for reactive modules with automatic conversion"""

    def __init__(self, ctx=None, names=None):
        """Initialize module
        
        Args:
            ctx: Context object for wire registry (optional for legacy use)
            names: Dictionary with 'extl', 'intf', 'prvt' keys mapping to lists of wire names
                   Example: {'extl': ['observation'], 'intf': ['q_values'], 'prvt': []}
        """
        if names is not None:
            # New API: auto-converting reactive module
            self.ctx = ctx
            self.extl = names.get('extl', [])
            self.intf = names.get('intf', [])
            self.prvt = names.get('prvt', [])
            
            # Will be set after conversion
            self._reactive_module = None
            self._wire_pairs = {}  # name -> (latched_wire, next_wire)
        else:
            # Legacy API: just wire declarations
            self.ctx = None
            self.extl = []
            self.intf = []
            self.prvt = []
            self._reactive_module = None
            self._wire_pairs = {}
        
    def _finalize_conversion(self):
        """Called automatically after subclass __init__ completes"""
        if self.ctx is not None:
            # Convert to reactive module
            self._reactive_module = convert_to_module(self.ctx, self)
            
            # Extract wire pairs from the converted module
            if hasattr(self._reactive_module, '_wire_pairs_dict'):
                self._wire_pairs = self._reactive_module._wire_pairs_dict
        
    @property
    def obs(self):
        """Observable wires: external inputs + interface outputs"""
        return self.extl + self.intf

    @property
    def ctrl(self):
        """Controlled wires: interface outputs + private wires"""
        return self.intf + self.prvt
    
    @property
    def intf_wires(self):
        """Get interface wires as list of (latched, next) wire pairs"""
        return [self._wire_pairs[name] for name in self.intf if name in self._wire_pairs]
    
    @property
    def intf_named(self):
        """Get interface as list of (name, (latched, next)) tuples"""
        return [(name, self._wire_pairs[name]) for name in self.intf if name in self._wire_pairs]
    
    @property
    def extl_wires(self):
        """Get external wires as list of (latched, next) wire pairs"""
        return [self._wire_pairs[name] for name in self.extl if name in self._wire_pairs]
    
    @property
    def extl_named(self):
        """Get external as list of (name, (latched, next)) tuples"""
        return [(name, self._wire_pairs[name]) for name in self.extl if name in self._wire_pairs]
    
    @property
    def prvt_wires(self):
        """Get private wires as list of (latched, next) wire pairs"""
        return [self._wire_pairs[name] for name in self.prvt if name in self._wire_pairs]
    
    @property
    def prvt_named(self):
        """Get private as list of (name, (latched, next)) tuples"""
        return [(name, self._wire_pairs[name]) for name in self.prvt if name in self._wire_pairs]

    def init(self, **inputs):
        """Initialize module state"""
        raise NotImplementedError

    def update(self, **inputs):
        """Update module (one tick/step)"""
        raise NotImplementedError
    
    def to_rust_module(self):
        """Convert this Python module to Rust Module representation
        
        This will be implemented later to bridge to zrth._zrth.toy.Module
        """
        raise NotImplementedError("Rust bridge not yet implemented")
    
    def __init_subclass__(cls, **kwargs):
        """Hook to automatically call conversion after subclass __init__"""
        super().__init_subclass__(**kwargs)
        
        # Save original __init__
        original_init = cls.__init__
        
        def wrapped_init(self, *args, **kwargs):
            # Call original __init__
            original_init(self, *args, **kwargs)
            
            # Auto-convert if this is a Module instance with context
            if hasattr(self, '_finalize_conversion') and hasattr(self, 'ctx') and self.ctx is not None:
                self._finalize_conversion()
        
        # Replace __init__ with wrapped version
        cls.__init__ = wrapped_init
