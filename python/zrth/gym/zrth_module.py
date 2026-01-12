from .converter import convert_to_module
from .context import Context


class Module:
    """Base class for reactive modules with automatic conversion"""
    
    _global_context = None
    
    @classmethod
    def get_global_context(cls):
        """Get or create the shared global context"""
        if cls._global_context is None:
            cls._global_context = Context()
        return cls._global_context

    def __init__(self, ctx=None, names=None):
        """Initialize module
        
        Args:
            ctx: Context object for wire registry (if None, uses global shared context)
            names: Dictionary with 'extl', 'intf', 'prvt' keys mapping to lists of wire names
                   Example: {'extl': ['observation'], 'intf': ['q_values'], 'prvt': []}
        """
        self.ctx = ctx if ctx is not None else Module.get_global_context()
        self.extl = names.get('extl', []) if names else []
        self.intf = names.get('intf', []) if names else []
        self.prvt = names.get('prvt', []) if names else []
        
        # Will be set after conversion
        self._reactive_module = None
        
    def _finalize_conversion(self):
        """Called automatically after subclass __init__ completes"""
        if self.ctx is not None:
            # Convert to reactive module
            self._reactive_module = convert_to_module(self.ctx, self)
        
    @property
    def obs(self):
        """Observable wires: external inputs + interface outputs"""
        return self.extl + self.intf

    @property
    def ctrl(self):
        """Controlled wires: interface outputs + private wires"""
        return self.intf + self.prvt

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
