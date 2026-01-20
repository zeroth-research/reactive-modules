from zrth import _zrth
from .torch.ll import DType, Wire

from torch import Tensor
from typing import Callable


class Context:
    """
    Context object used to hold information about known variables,
    their mapping to wire identifiers, and created terms.

    This class is the super-class for specific Context classes of different
    crates. See, e.g., :class:`toy.Context`.
    """

    def __init__(self, rust_ctx=_zrth.torch.RustContext()):
        """
        :param: ctx_impl  is the Rust context object.
        """
        self._rust_context = rust_ctx
        # when we are tracing code, we create terms and we store them
        # in lists that are here. The lists form a stack,
        # the user can push a new frame (list) and pop an old one
        # to distinguish terms during tracing different parts of code
        self._terms_frames = []

    def push_terms_frame(self, f: list) -> None:
        self._terms_frames.append(f)

    def pop_terms_frame(self) -> list:
        return self._terms_frames.pop()

    def add_term(self, term):
        if self._terms_frames:
            self._terms_frames[-1].append(term)

    def unwrap(self):
        return self._rust_context

    def tmp_wire(self, dtype: DType) -> Wire:
        return self.unwrap().tmp_wire(dtype)
