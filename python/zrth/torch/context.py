from zrth import _zrth
from zrth.context import Context as ContextBase
from .ll import DType, Wire

from torch import Tensor
from typing import Callable


Module = _zrth.torch.Module


class Context(ContextBase):
    def __init__(self):
        super().__init__(_zrth.torch.RustContext())

    def tmp_wire(self, dtype: DType) -> Wire:
        return self.unwrap().tmp_wire(dtype)
