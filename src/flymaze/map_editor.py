"""Local map editor and JSON validator; run with python -m flymaze.map_editor."""

from __future__ import annotations

import argparse
import json
import secrets
import tempfile
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from flymaze.custom_maps import GridMapSpec, read_grid_map

ASSETS = Path(__file__).with_name("editor_assets")


def make_handler(directory: Path, token: str) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code: int, value: Any) -> None:
            data = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def authorized(self) -> bool:
            supplied = self.headers.get("X-FlyMaze-Token", "")
            port = self.server.server_address[1]
            expected_host = f"127.0.0.1:{port}"
            return self.headers.get("Host") in (
                expected_host, f"localhost:{port}"
            ) and secrets.compare_digest(supplied, token)

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            if url.path == "/api/maps":
                if not self.authorized():
                    self.reply(403, {"error": "unauthorized"})
                    return
                maps, errors = [], []
                for path in sorted(directory.glob("custom_*.json")):
                    try:
                        maps.append(read_grid_map(path).to_dict())
                    except (ValueError, OSError) as exc:
                        errors.append(f"{path.name}: {exc}")
                self.reply(200, {"maps": maps, "errors": errors})
                return
            files = {"/": "index.html", "/editor.js": "editor.js", "/editor.css": "editor.css"}
            filename = files.get(url.path)
            if filename is None:
                self.reply(404, {"error": "not found"})
                return
            if url.path == "/" and not secrets.compare_digest(
                parse_qs(url.query).get("token", [""])[0], token
            ):
                self.reply(403, {"error": "open the complete URL printed in the terminal"})
                return
            data = (ASSETS / filename).read_bytes()
            self.send_response(200)
            mime = {"html": "text/html", "js": "text/javascript", "css": "text/css"}
            self.send_header("Content-Type", mime[filename.rsplit(".", 1)[1]] + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            if not self.authorized():
                self.reply(403, {"error": "unauthorized"})
                return
            if self.path not in ("/api/check", "/api/save"):
                self.reply(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 100_000:
                    raise ValueError("request must contain at most 100 KB of JSON")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("request must be an object")
                spec = GridMapSpec.from_dict(payload.get("map"))
                path = directory / f"{spec.map_id}.json"
                if self.path == "/api/save":
                    if path.exists() and payload.get("overwrite") is not True:
                        self.reply(409, {"error": "a map with this ID already exists"})
                        return
                    directory.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(
                        mode="w", encoding="utf-8", dir=directory, delete=False, suffix=".tmp"
                    ) as stream:
                        temporary = Path(stream.name)
                        json.dump(spec.to_dict(), stream, ensure_ascii=False, indent=2)
                        stream.write("\n")
                    try:
                        temporary.replace(path)
                    finally:
                        temporary.unlink(missing_ok=True)
                self.reply(200, {
                    "ok": True, "path": str(path), "map_id": spec.map_id,
                    "route_length_mm": (len(spec.shortest_path()) - 1) * 10,
                    "wall_count": len(spec.wall_boxes()),
                })
            except (ValueError, OSError) as exc:
                self.reply(400, {"error": str(exc)})

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyMaze map editor (standard library only)")
    parser.add_argument("--map-directory", type=Path, default=Path("maps"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--check", type=Path, help="validate one JSON map without opening the editor"
    )
    args = parser.parse_args()
    if args.check:
        spec = read_grid_map(args.check)
        print(json.dumps({"valid": True, "map": spec.to_dict(),
                          "route_length_mm": 10 * (len(spec.shortest_path()) - 1)}, indent=2))
        return
    token = secrets.token_urlsafe(24)
    directory = args.map_directory.resolve()
    with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(directory, token)) as server:
        url = f"http://127.0.0.1:{server.server_port}/?token={token}"
        print(
            f"FlyMaze map editor: {url}\nSave directory: {directory}\nCtrl+C to stop.",
            flush=True,
        )
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
