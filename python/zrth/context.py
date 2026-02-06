from .zrth import DType, Wire, RustContext


class ContextBase:
    """
    Context object used to hold information about known variables,
    (their mapping to wire identifiers), and, mainly, created terms.

    This class is the super-class for specific `Context` class, but will
    be probably merged with that class.
    At this moment it is kept separately to separate terms tracking
    and ID creation from the rest (e.g., names bookkeeping).

    XXX: we currently do not use named wires with `RustContext`
    """

    def __init__(self, rust_ctx=None):
        """
        :param: ctx_impl  is the Rust context object.
        """
        self._rust_context = rust_ctx or RustContext()
        # when we are tracing code, we create terms and we store them
        # in lists that are here. The lists form a stack,
        # the user can push a new frame (list) and pop an old one
        # to distinguish terms during tracing different parts of code
        self._terms_frames = []

        self.uninterpreted = {}

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

    def declare_const(self, name: str, dtype: DType):
        if name in self.uninterpreted:
            raise KeyError(f"Uninterpreted function/constant '{name}' already exists")
        self.uninterpreted[name] = dtype

    def get_dtype(self, name: str):
        try:
            return self.uninterpreted[name]
        except KeyError:
            raise KeyError(f"Uninterpreted function/constant '{name}' does not exists") from None


class Context(ContextBase):
    """Wire registry for building reactive modules

    Manages wire ID generation and name-to-ID mapping.
    Follows the common-ctx API pattern for future compatibility.
    """

    def __init__(self, rust_ctx=None):
        super().__init__(rust_ctx)
        self.name_to_id: dict[str, int] = {}  # str -> int
        self.id_to_name: dict[int, str] = {}  # int -> str
        self._wire_dtypes: dict[int, DType] = {}  # id -> DType enum

    def fresh_wire_id(self) -> int:
        """Create unnamed temporary wire ID

        Returns:
            Wire ID (int)
        """
        v = self.unwrap().fresh_wire_id()
        name = f"__c_{v}"
        self.name_to_id[name] = v
        self.id_to_name[v] = name
        return v

    def get_wire_id(self, name: str) -> int:
        """Get or create named wire ID

        Args:
            name: Wire name

        Returns:
            Wire ID (int)
        """
        assert isinstance(name, str), (name, type(name))

        if name not in self.name_to_id:
            wid = self.unwrap().fresh_wire_id()
            self.name_to_id[name] = wid
            self.id_to_name[wid] = name
        return self.name_to_id[name]

    def wire(self, name: str, dtype: DType) -> Wire:
        """Get or create named wire with dtype

        Args:
            name: Wire name
            dtype: Datatype (string like 'Tensor', 'Bool')

        Returns:
            Wire object
        """
        assert isinstance(name, str), (name, type(name))
        assert isinstance(dtype, DType), (dtype, type(dtype))

        wire_id = self.get_wire_id(name)
        self._wire_dtypes[wire_id] = dtype
        return Wire(wire_id, dtype)

    def tmp_wire(self, dtype: DType) -> Wire:
        """Create temporary wire

        Args:
            dtype: Datatype (string like 'Tensor', 'Bool')

        Returns:
            Wire object
        """
        assert isinstance(dtype, DType), (dtype, type(dtype))

        wire_id = self.fresh_wire_id()
        self._wire_dtypes[wire_id] = dtype
        return Wire(wire_id, dtype)

    def has_wire(self, name: str) -> bool:
        """Check if a wire is declared

        Args:
            name: Wire name

        Returns:
            True if wire exists
        """
        assert isinstance(name, str), (name, type(name))
        return name in self.name_to_id

    def num_wires(self) -> int:
        """Get number of created wires

        Returns:
            Number of wires
        """
        return len(self.unwrap())

    def get_name(self, wire_id: int) -> str | None:
        """Get name for a wire ID (if it has one)

        Args:
            wire_id: Wire ID

        Returns:
            Name or None
        """
        assert isinstance(wire_id, int), (wire_id, type(wire_id))
        return self.id_to_name.get(wire_id)

    def __str__(self) -> str:
        result = f"Context with {self.num_wires()} wires:\n"
        for name, wid in sorted(self.name_to_id.items(), key=lambda x: x[1]):
            dtype = self._wire_dtypes[wid]
            result += f"  {name} (id={wid}): {dtype}\n"
        return result


#####################################################################
# Global context
#####################################################################

# Term are created in this global context.
# The context can be switched manually, but that is for advanced users
_global_context = Context()


def set_ctx(ctx: Context) -> Context:
    global _global_context
    old = _global_context
    _global_context = ctx
    return old


def get_ctx() -> Context:
    global _global_context
    return _global_context


def reset_ctx() -> Context:
    global _global_context
    _global_context = Context()
    return _global_context
