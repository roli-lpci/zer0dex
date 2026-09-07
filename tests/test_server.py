"""Tests for zer0dex server — handler logic without requiring mem0/Ollama."""
import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# We can't import server.py at module level because it imports mem0.
# Instead, test the handler class by patching mem0 before import.
mock_mem0 = MagicMock()
with patch.dict("sys.modules", {"mem0": mock_mem0}):
    from zer0dex.server import Mem0Handler


class FakeRequest(BytesIO):
    def makefile(self, *args, **kwargs):
        return self


def make_handler(method, path, body=None):
    """Create a Mem0Handler with mocked socket/request."""
    if body:
        body_bytes = json.dumps(body).encode()
    else:
        body_bytes = b""

    handler = Mem0Handler.__new__(Mem0Handler)
    handler.rfile = BytesIO(body_bytes)
    handler.wfile = BytesIO()
    handler.headers = {"Content-Length": str(len(body_bytes)), "Content-Type": "application/json"}
    handler.path = path
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = method
    handler.client_address = ("127.0.0.1", 9999)

    # Capture response
    handler._response_code = None
    handler._response_headers = {}
    handler._response_body = None

    def capture_send_json(data, status=200):
        handler._response_code = status
        handler._response_body = data

    handler._send_json = capture_send_json

    return handler


class TestReadBody:
    def test_reads_json(self):
        handler = make_handler("POST", "/query", {"text": "hello"})
        # Restore real _read_body
        handler._read_body = Mem0Handler._read_body.__get__(handler, Mem0Handler)
        result = handler._read_body()
        assert result == {"text": "hello"}

    def test_empty_body(self):
        handler = make_handler("POST", "/query")
        handler.headers = {"Content-Length": "0"}
        handler.rfile = BytesIO(b"")
        handler._read_body = Mem0Handler._read_body.__get__(handler, Mem0Handler)
        result = handler._read_body()
        assert result == {}

    def test_invalid_json(self):
        handler = make_handler("POST", "/query")
        handler.headers = {"Content-Length": "5"}
        handler.rfile = BytesIO(b"notjs")
        handler._read_body = Mem0Handler._read_body.__get__(handler, Mem0Handler)
        result = handler._read_body()
        assert result is None


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        handler = make_handler("GET", "/health")
        mock_memory = MagicMock()
        mock_memory.get_all.return_value = {"results": [{"memory": "a"}, {"memory": "b"}]}
        handler.memory = mock_memory
        handler.user_id = "agent"

        handler.do_GET()
        assert handler._response_code == 200
        assert handler._response_body["status"] == "ok"
        assert handler._response_body["count"] == 2
        mock_memory.get_all.assert_called_once_with(
            filters={"user_id": "agent"}, top_k=100
        )

    def test_unknown_get_returns_404(self):
        handler = make_handler("GET", "/unknown")
        handler.do_GET()
        assert handler._response_code == 404

    def test_lifecycle_requires_exact_launch_token(self, monkeypatch):
        handler = make_handler("GET", "/_lifecycle")
        handler.instance_token = "launch-token"
        handler.headers["X-Zer0dex-Instance-Token"] = "launch-token"
        monkeypatch.setattr("zer0dex.server.os.getpid", lambda: 123)

        handler.do_GET()

        assert handler._response_code == 200
        assert handler._response_body == {"pid": 123}

    @pytest.mark.parametrize("token", ["", "wrong-token"])
    def test_lifecycle_rejects_missing_or_wrong_token(self, token):
        handler = make_handler("GET", "/_lifecycle")
        handler.instance_token = "launch-token"
        if token:
            handler.headers["X-Zer0dex-Instance-Token"] = token

        handler.do_GET()

        assert handler._response_code == 404


class TestQueryEndpoint:
    def test_query_filters_by_min_score(self):
        handler = make_handler("POST", "/query", {"text": "test query", "limit": 5})
        handler._read_body = lambda: {"text": "test query", "limit": 5}
        mock_memory = MagicMock()
        mock_memory.search.return_value = {
            "results": [
                {"memory": "good", "score": 0.8},
                {"memory": "bad", "score": 0.1},
            ]
        }
        handler.memory = mock_memory
        handler.user_id = "agent"
        handler.min_score = 0.3

        handler.do_POST()
        assert handler._response_code == 200
        assert len(handler._response_body["memories"]) == 1
        assert handler._response_body["memories"][0]["text"] == "good"
        mock_memory.search.assert_called_once_with(
            "test query", filters={"user_id": "agent"}, top_k=5
        )

    def test_query_empty_text_returns_empty(self):
        handler = make_handler("POST", "/query", {"text": ""})
        handler._read_body = lambda: {"text": ""}
        handler.path = "/query"
        handler.memory = MagicMock()
        handler.user_id = "agent"
        handler.min_score = 0.3

        handler.do_POST()
        assert handler._response_body["memories"] == []

    def test_query_short_text_returns_empty(self):
        handler = make_handler("POST", "/query", {"text": "ab"})
        handler._read_body = lambda: {"text": "ab"}
        handler.path = "/query"
        handler.memory = MagicMock()
        handler.user_id = "agent"
        handler.min_score = 0.3

        handler.do_POST()
        assert handler._response_body["memories"] == []


class TestAddEndpoint:
    def test_add_returns_count(self):
        handler = make_handler("POST", "/add", {"text": "new memory"})
        handler._read_body = lambda: {"text": "new memory"}
        handler.path = "/add"
        mock_memory = MagicMock()
        mock_memory.add.return_value = {"results": [{"memory": "new memory"}]}
        handler.memory = mock_memory
        handler.user_id = "agent"

        handler.do_POST()
        assert handler._response_code == 200
        assert handler._response_body["count"] == 1

    def test_add_empty_text_returns_400(self):
        handler = make_handler("POST", "/add", {"text": ""})
        handler._read_body = lambda: {"text": ""}
        handler.path = "/add"
        handler.memory = MagicMock()
        handler.user_id = "agent"

        handler.do_POST()
        assert handler._response_code == 400

    def test_unknown_post_returns_404(self):
        handler = make_handler("POST", "/unknown", {})
        handler._read_body = lambda: {}
        handler.path = "/unknown"
        handler.memory = MagicMock()
        handler.user_id = "agent"

        handler.do_POST()
        assert handler._response_code == 404

    def test_invalid_json_returns_400(self):
        handler = make_handler("POST", "/add")
        handler._read_body = lambda: None
        handler.path = "/add"
        handler.memory = MagicMock()
        handler.user_id = "agent"

        handler.do_POST()
        assert handler._response_code == 400

    def test_json_array_returns_400(self):
        handler = make_handler("POST", "/query", {})
        handler._read_body = lambda: ["not", "an", "object"]
        handler.path = "/query"

        handler.do_POST()
        assert handler._response_code == 400
        assert handler._response_body == {"error": "invalid json"}

    def test_query_rejects_invalid_limit(self):
        handler = make_handler("POST", "/query", {"text": "test query", "limit": 0})
        handler._read_body = lambda: {"text": "test query", "limit": 0}
        handler.path = "/query"
        handler.memory = MagicMock()

        handler.do_POST()
        assert handler._response_code == 400
        assert handler._response_body == {"error": "limit must be a positive integer"}

    def test_add_rejects_non_string_text(self):
        handler = make_handler("POST", "/add", {"text": 1})
        handler._read_body = lambda: {"text": 1}
        handler.path = "/add"
        handler.memory = MagicMock()

        handler.do_POST()
        assert handler._response_code == 400
        assert handler._response_body == {"error": "text must be a non-empty string"}
