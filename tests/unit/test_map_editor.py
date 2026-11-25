"""Exercise the real HTTP handlers in memory, including file writes and error responses."""

import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from test_custom_maps import example

from flymaze.map_editor import make_handler


class EditorHandlerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def request(
        self, method: str, path: str, payload: Any = None,
        token: str = "test-token", host: str = "localhost:8765",
    ) -> tuple[int, bytes]:
        handler: Any = object.__new__(make_handler(self.directory, "test-token"))
        data = json.dumps(payload).encode() if payload is not None else b""
        handler.path = path
        handler.server = SimpleNamespace(server_address=("127.0.0.1", 8765))
        handler.headers = Message()
        handler.headers["Host"] = host
        handler.headers["X-FlyMaze-Token"] = token
        handler.headers["Content-Length"] = str(len(data))
        handler.rfile = io.BytesIO(data)
        handler.wfile = io.BytesIO()
        result: dict[str, int] = {}
        handler.send_response = lambda status: result.update(status=status)
        handler.send_header = lambda *args: None
        handler.end_headers = lambda: None
        getattr(handler, f"do_{method}")()
        return result["status"], handler.wfile.getvalue()

    def test_check_save_load_and_explicit_overwrite(self) -> None:
        status, body = self.request("POST", "/api/check", {"map": example()})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["route_length_mm"], 40)
        self.assertFalse(list(self.directory.iterdir()))
        self.assertEqual(self.request("POST", "/api/save", {"map": example()})[0], 200)
        self.assertEqual(self.request("POST", "/api/save", {"map": example()})[0], 409)
        status, body = self.request("GET", "/api/maps")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["maps"][0]["map_id"], "custom_left")
        self.assertEqual(self.request("POST", "/api/save", {
            "map": example() | {"timeout_seconds": 20}, "overwrite": True,
        })[0], 200)
        self.assertFalse(list(self.directory.glob("*.tmp")))

    def test_validation_authentication_and_no_path_traversal(self) -> None:
        self.assertEqual(self.request("GET", "/api/maps", token="wrong")[0], 403)
        self.assertEqual(self.request("POST", "/api/save", {"map": example()},
                                      host="attacker.example")[0], 403)
        self.assertEqual(self.request("GET", "/../../pyproject.toml")[0], 404)
        self.assertEqual(self.request("POST", "/api/save", {
            "map": example() | {"map_id": "../../evil"},
        })[0], 400)
        self.assertFalse(list(self.directory.iterdir()))

    def test_static_assets_and_partial_bad_map_listing(self) -> None:
        self.assertEqual(self.request("GET", "/")[0], 403)
        for path in ("/?token=test-token", "/editor.js", "/editor.css"):
            status, body = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertTrue(body)
        (self.directory / "custom_broken.json").write_text("bad", encoding="utf-8")
        status, body = self.request("GET", "/api/maps")
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["errors"]), 1)


if __name__ == "__main__":
    unittest.main()
