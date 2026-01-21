from .converter import convert_to_module
from .. import Context, get_ctx


class Module:
    """Base class for reactive modules with automatic conversion"""

    def __init__(self, extl, intf, prvt=None, ctx=None):
        """Initialize module

        Args:
            ctx: Context object for wire registry (if None, uses global shared context)
            extl: List of external input wire names
            intf: List of interface output wire names
            prvt: List of private wire names (optional)
        """
        self.ctx = ctx if ctx is not None else get_ctx()
        self.extl = extl
        self.intf = intf
        self.prvt = prvt

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
        return self.intf + (self.prvt if self.prvt is not None else [])

    def __init_subclass__(cls, **kwargs):
        """Hook to automatically call conversion after subclass __init__"""
        super().__init_subclass__(**kwargs)

        # Save original __init__
        original_init = cls.__init__

        def wrapped_init(self, *args, **kwargs):
            # Call original __init__
            original_init(self, *args, **kwargs)

            # Auto-convert if this is a Module instance with context
            if (
                hasattr(self, "_finalize_conversion")
                and hasattr(self, "ctx")
                and self.ctx is not None
            ):
                self._finalize_conversion()

        # Replace __init__ with wrapped version
        cls.__init__ = wrapped_init
