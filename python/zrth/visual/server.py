"""Live module visualizer with auto-updating via WebSocket.

Usage:
    from zrth.visual import show
    show(module)           # opens browser, auto-refreshes on value changes
    show(module, poll=0.5) # custom poll interval in seconds
"""
import json
import socket
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import asyncio
import websockets


_TEMPLATE_PATH = Path(__file__).parent / "template.html"


def _serialize_terms(terms, atom_idx, phase, counter):
    """Serialize a block of terms to JSON-compatible dicts."""
    result = []
    for term in terms:
        tid = f"a{atom_idx}_{phase}_{counter[0]}"
        counter[0] += 1
        writes = list(term.write)
        result.append({
            "id": tid,
            "label": str(term.itype),
            "reads": [w.id for w in term.read],
            "writes": [w.id for w in writes],
            "dtype": str(writes[0].dtype) if writes else None,
        })
    return result


def _serialize_module(module):
    """Serialize a Module to a JSON-compatible dict."""
    counter = [0]
    atoms = []

    for atom_idx, atom in enumerate(module.atoms):
        atoms.append({
            "id": atom_idx,
            "ctrl": [{"id": w.id, "dtype": str(w.dtype)} for w in atom.ctrl],
            "wait": [{"id": w.id, "dtype": str(w.dtype)} for w in atom.wait],
            "read": [{"id": w.id, "dtype": str(w.dtype)} for w in atom.read],
            "init": _serialize_terms(atom.init, atom_idx, "init", counter),
            "update": _serialize_terms(atom.update, atom_idx, "update", counter),
        })

    def wire_pairs(iface):
        return [{"ltc": ltc.id, "nxt": nxt.id, "dtype": str(ltc.dtype)} for ltc, nxt in iface]

    return {
        "atoms": atoms,
        "extl": wire_pairs(module.extl),
        "intf": wire_pairs(module.intf),
        "prvt": wire_pairs(module.prvt),
    }


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def show(module, names=None, poll=0.5, open_browser=True):
    """Start a live visualizer for the given module.

    Args:
        module: A zrth Module (or Env/torch.Module instance).
        names: Optional dict mapping wire IDs (int) to display names (str).
               If None, circles show no labels (topology only).
        poll: Interval in seconds between state pushes.
        open_browser: Whether to open the browser automatically.

    Returns:
        A stop function that shuts down the server.
    """
    wire_names = {}
    if names:
        for k, v in names.items():
            if isinstance(k, (list, tuple)) and len(k) == 2:
                # Wire pair: latched gets name, next gets name'
                wire_names[k[0].id if hasattr(k[0], 'id') else int(k[0])] = str(v)
                wire_names[k[1].id if hasattr(k[1], 'id') else int(k[1])] = str(v) + "'"
            else:
                wire_names[k.id if hasattr(k, 'id') else int(k)] = str(v)
    http_port = _find_free_port()
    ws_port = _find_free_port()

    html = _TEMPLATE_PATH.read_text().replace("{{WS_PORT}}", str(ws_port))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        def log_message(self, format, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", http_port), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    stop_event = threading.Event()

    async def ws_handler(websocket):
        while not stop_event.is_set():
            data = _serialize_module(module)
            data['wire_names'] = wire_names
            try:
                await websocket.send(json.dumps({"type": "full", "data": data}))
            except websockets.exceptions.ConnectionClosed:
                break
            await asyncio.sleep(poll)

    async def ws_main():
        async with websockets.serve(ws_handler, "127.0.0.1", ws_port):
            while not stop_event.is_set():
                await asyncio.sleep(0.1)

    def run_ws():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ws_main())
        loop.close()

    threading.Thread(target=run_ws, daemon=True).start()

    url = f"http://127.0.0.1:{http_port}"
    if open_browser:
        webbrowser.open(url)
    print(f"Module viewer running at {url} (ws://127.0.0.1:{ws_port})")

    def stop():
        stop_event.set()
        httpd.shutdown()
        print("Module viewer stopped.")

    return stop
