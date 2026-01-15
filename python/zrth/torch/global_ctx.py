from .context import Context

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
