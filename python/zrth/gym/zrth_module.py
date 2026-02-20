from zrth.module import ReactiveModuleDef
from .converter import convert_to_module


class Module(ReactiveModuleDef):
    """Base class for reactive modules with automatic conversion
    
    Subclasses must declare wires as class attributes:
    - extl: List of external input wire names
    - intf: List of interface output wire names
    - prvt: List of private wire names (optional)
    """

    def __new__(cls, *args, **kwargs):
        """Create instance and automatically convert to reactive Module
        
        This method:
        1. Creates instance of subclass
        2. Gets wire declarations from class attributes
        3. Initializes ReactiveModuleDef with wires
        4. Calls subclass __init__ to set up layers/methods
        5. Converts to reactive Module and returns it
        """
        # Create instance of the subclass
        instance = object.__new__(cls)
        
        # Get wire declarations from class attributes
        extl = cls.__dict__.get('extl', [])
        intf = cls.__dict__.get('intf', [])
        prvt = cls.__dict__.get('prvt', None)
        
        # Delete class attributes to prevent shadowing ReactiveModuleDef properties
        for attr in ['extl', 'intf', 'prvt']:
            if attr in cls.__dict__:
                delattr(cls, attr)
        
        # Initialize ReactiveModuleDef with wires
        ReactiveModuleDef.__init__(instance, intf, extl, prvt)
        
        # Call subclass __init__ to set up layers, methods, etc.
        # This is where nn.Linear layers get created for networks
        # or reset/step methods are defined for environments
        cls.__init__(instance, *args, **kwargs)
        
        # Convert the fully initialized instance to reactive Module
        module = convert_to_module(instance)
        
        # Return the Rust Module instead of Python instance
        return module
