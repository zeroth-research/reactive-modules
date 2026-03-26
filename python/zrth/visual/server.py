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
        result.append({
            "id": tid,
            "label": str(term.itype),
            "reads": [w.id for w in term.read],
            "writes": [w.id for w in term.write],
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

    def wire_info(pairs):
        result = []
        for ltc, nxt in pairs:
            result.append({"id": ltc.id, "dtype": str(ltc.dtype)})
            result.append({"id": nxt.id, "dtype": str(nxt.dtype)})
        return result

    return {
        "atoms": atoms,
        "externals": wire_info(module.extl),
        "interfaces": wire_info(module.intf),
    }


def _find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def show(module, poll=0.5, open_browser=True):
    """Start a live visualizer for the given module.

    Args:
        module: A zrth Module (or Wrapper/Env/torch.Module instance).
        poll: Interval in seconds between state pushes.
        open_browser: Whether to open the browser automatically.

    Returns:
        A stop function that shuts down the server.
    """
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
