"""Live visualisation server for reactive modules.

Usage::

    import zrth.visual
    zrth.visual.start()          # opens http://localhost:7777 in the browser

    from gym.environments import SimpleEnv   # graph appears automatically

    env = SimpleEnv()
    env.reset()
    env.step(tensor)             # wire values update in the browser
"""


def start(port: int = 7777) -> None:
    """Start the HTTP+WebSocket visualisation server and open a browser tab.

    Calling this more than once is a no-op — the server starts at most once per
    process.  The default port is 7777.
    """
    try:
        from zrth.zrth import _visual_start
    except ImportError:
        raise ImportError(
            "The visual server is not available.  "
            "Rebuild the extension with the 'visual-server' Cargo feature enabled."
        )
    _visual_start(port)


def push(module) -> None:
    """Manually push a module graph to the browser.

    Normally this is called automatically when an Env or NN subclass is
    instantiated, but you can call it explicitly if needed.
    """
    try:
        from zrth.zrth import _visual_push_module
        _visual_push_module(module)
    except Exception:
        pass
