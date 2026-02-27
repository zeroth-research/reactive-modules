"""Live reactive-module visualiser — file-watch entry point.

Opens a browser at http://localhost:7777 and keeps it in sync with
module_def.py.  Edit that file, save it, and the graph updates instantly.

Run with:
    cd python && uv run python scripts/visual/ui_watch.py

Workflow:
    1. Run the command above — the browser opens automatically.
    2. Open scripts/visual/module_def.py in your editor.
    3. Edit the module (wires, terms, constants) and save.
    4. The browser graph updates within ~0.3 s.
    5. Press Ctrl-C in the terminal to stop.
"""
import os
import zrth.visual

_here = os.path.dirname(os.path.abspath(__file__))
zrth.visual.watch(os.path.join(_here, "module_def.py"))
