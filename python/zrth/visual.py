"""Live visualisation server for reactive modules.

Usage::

    import zrth.visual
    zrth.visual.start()          # opens http://localhost:7777 in the browser

    from gym.environments import SimpleEnv   # graph appears automatically

    env = SimpleEnv()
    env.reset()
    env.step(tensor)             # wire values update in the browser

File-watch workflow::

    # scripts/visual/ui_watch.py
    import zrth.visual
    zrth.visual.watch("module_def.py")   # blocks; edit & save to update browser
"""
import os
import time


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


def _exec_file(path: str) -> None:
    """Execute *path* in a fresh namespace, printing any exception."""
    ns: dict = {}
    with open(path) as fh:
        code = compile(fh.read(), path, "exec")
    try:
        exec(code, ns)  # noqa: S102
    except Exception as exc:
        print(f"[watch] Error in {os.path.basename(path)}: {exc}")


def watch(path: str, port: int = 7777) -> None:
    """Start the server, execute *path* once, then block watching for changes.

    Each time the file is saved the server re-executes it so the browser
    graph updates live.  Press Ctrl-C to stop.

    The file is run in its own fresh namespace on every reload — any call to
    ``zrth.visual.push(module)`` inside it pushes the new graph to the browser.
    """
    start(port)

    path = os.path.abspath(path)
    _exec_file(path)
    mtime = os.path.getmtime(path)

    print(f"Watching {path}")
    print("Edit and save the file to update the browser.  Ctrl-C to stop.\n")
    try:
        while True:
            time.sleep(0.3)
            try:
                new_mtime = os.path.getmtime(path)
            except OSError:
                continue
            if new_mtime != mtime:
                mtime = new_mtime
                print(f"[watch] reloading {os.path.basename(path)} …")
                _exec_file(path)
    except KeyboardInterrupt:
        print("\n[watch] Stopped.")
