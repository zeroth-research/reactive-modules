class Context:
    """
    Context object used to hold information about known variables,
    their mapping to wire identifiers, and created terms.

    This class is the super-class for specific Context classes of different
    crates. See, e.g., :class:`toy.Context`.
    """

    def __init__(self, ctx_impl):
        """
        :param: ctx_impl  is the Rust context object.
        """
        self._context = ctx_impl
        # here we store terms during tracing a function
        # with the `trace` method.
        self._terms: list | None = None

    def unwrap(self):
        return self._context

   #def fresh_var_id(self) -> int:
   #    return self._context.fresh_var()
   #
   #def get_var_id(self, name: str) -> int:
   #    return self._context.get_var(name)

    def _start_gathering_terms(self) -> None:
        assert self._terms is None, "Already gathering terms"
        self._terms = []

    def _stop_gathering_terms(self) -> None:
        tmp = self._terms
        self._terms = None

        return tmp

    def add_traced_term(self, term) -> None:
        terms = self._terms
        if terms is not None:
            terms.append(term)
