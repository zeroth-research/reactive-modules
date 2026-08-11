from .zrth import Sort as _Sort
from abc import ABC


class Sort(ABC):
    """Sort type. Variants are available as module-level names."""


Sort.register(_Sort)

# makes sorts available on the base namespace
for _name in dir(_Sort):
    if not _name.startswith('_'):
        globals()[_name] = getattr(_Sort, _name)
