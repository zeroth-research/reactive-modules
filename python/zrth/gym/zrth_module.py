from zrth.module import ReactiveModuleDef, call_convert_after_init
from .converter import convert_to_module
from .. import Context, get_ctx


class Module(ReactiveModuleDef):
    """Base class for reactive modules with automatic conversion"""

    def __init__(self, extl, intf, prvt=None):
        """Initialize module

        Args:
            ctx: Context object for wire registry (if None, uses global shared context)
            extl: List of external input wire names
            intf: List of interface output wire names
            prvt: List of private wire names (optional)
        """
        super().__init__(intf, extl, prvt)

    def __init_subclass__(cls, **kwargs):
        """
        Hook to automatically call conversion after subclass' __init__.
        The conversion *must* be done after subclasses are fully initialized
        because the conversion uses data from the classes
        """
        call_convert_after_init(__class__, cls, **kwargs)

    def convert(self):
        """
        Called automatically after subclass __init__ completes.

        Convert self into a reactive module and save it in `self._module`
        """
        assert self._module is None

        self._module = convert_to_module(self)
