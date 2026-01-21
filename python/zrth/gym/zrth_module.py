from zrth.module import ReactiveModuleDef
from .converter import convert_to_module
from .. import Context, get_ctx


class Module(ReactiveModuleDef):
    """Base class for reactive modules with automatic conversion"""

    def __init__(self, extl, intf, prvt=None, ctx=None):
        """Initialize module

        Args:
            ctx: Context object for wire registry (if None, uses global shared context)
            extl: List of external input wire names
            intf: List of interface output wire names
            prvt: List of private wire names (optional)
        """
        super().__init__(intf, extl, prvt, ctx=ctx)

    def __init_subclass__(cls, **kwargs):
        """
        Hook to automatically call conversion after subclass' __init__.
        The conversion *must* be done after subclasses are fully initialized
        because the conversion uses data from the classes
        """
        super().__init_subclass__(**kwargs)

        # Save original __init__
        original_init = cls.__init__

        # Wrap the init to call `convert` after the original init finishes
        def wrapped_init(self, *args, **kwargs):
            # Call original __init__
            original_init(self, *args, **kwargs)

            self.convert()

        # Replace __init__ with wrapped version
        cls.__init__ = wrapped_init

    def convert(self):
        """
        Called automatically after subclass __init__ completes.

        Convert self into a reactive module and save it in `self._module`
        """
        assert self._module is None
        assert self._ctx is not None

        self._module = convert_to_module(self._ctx, self)
