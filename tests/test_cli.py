"""Tests for zer0dex CLI — config, parsing, and check command."""
import argparse
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from zer0dex import cli
from zer0dex.cli import load_config, positive_int, save_config
from zer0dex.server import Mem0Handler


@pytest.fixture
def tmp_workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestConfig:
    def test_load_config_missing_file(self, tmp_workdir):
        assert load_config() == {}

    def test_save_and_load_roundtrip(self, tmp_workdir):
        cfg = {"collection": "test", "port": 9999}
        save_config(cfg)
        loaded = load_config()
        assert loaded == cfg

    def test_save_creates_json_file(self, tmp_workdir):
        save_config({"key": "val"})
        raw = (tmp_workdir / ".zer0dex.json").read_text()
        assert json.loads(raw) == {"key": "val"}


class TestCLIParsing:
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, "-m", "zer0dex.cli", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")},
        )
        assert result.returncode == 0
        assert "zer0dex" in result.stdout

    def test_no_command_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "zer0dex.cli"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")},
        )
        assert result.returncode == 1

    def test_check_subcommand_in_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "zer0dex.cli", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")},
        )
        assert "check" in result.stdout

    def test_all_commands_in_help(self):
        result = subprocess.run(
            [sys.executable, "-m", "zer0dex.cli", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")},
        )
        for cmd in ["check", "init", "seed", "serve", "stop", "query", "status", "add"]:
            assert cmd in result.stdout, f"Missing command: {cmd}"


class TestInit:
    def test_init_creates_config_and_dir(self, tmp_workdir):
        result = subprocess.run(
            [sys.executable, "-m", "zer0dex.cli", "init"],
            capture_output=True, text=True,
            cwd=str(tmp_workdir),
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")},
        )
        assert result.returncode == 0
        assert (tmp_workdir / ".zer0dex.json").exists()
        assert (tmp_workdir / ".zer0dex").is_dir()

    def test_init_config_has_required_keys(self, tmp_workdir):
        subprocess.run(
            [sys.executable, "-m", "zer0dex.cli", "init"],
            capture_output=True, text=True,
            cwd=str(tmp_workdir),
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")},
        )
        cfg = json.loads((tmp_workdir / ".zer0dex.json").read_text())
        for key in ["collection", "chroma_path", "port", "user_id", "llm_model", "embed_model", "ollama_url"]:
            assert key in cfg, f"Missing config key: {key}"

    def test_init_custom_collection(self, tmp_workdir):
        subprocess.run(
            [sys.executable, "-m", "zer0dex.cli", "init", "--collection", "mytest"],
            capture_output=True, text=True,
            cwd=str(tmp_workdir),
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")},
        )
        cfg = json.loads((tmp_workdir / ".zer0dex.json").read_text())
        assert cfg["collection"] == "mytest"


class TestCheck:
    def test_check_fails_without_ollama(self, tmp_workdir):
        """check should exit 1 when Ollama isn't running."""
        save_config({"ollama_url": "http://localhost:99999"})
        result = subprocess.run(
            [sys.executable, "-m", "zer0dex.cli", "check"],
            capture_output=True, text=True,
            cwd=str(tmp_workdir),
            env={**__import__("os").environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent / "src")},
        )
        assert result.returncode == 1
        assert "not reachable" in result.stdout or "not reachable" in result.stderr

    def test_check_requires_python_ollama_client(self, tmp_workdir, monkeypatch, capsys):
        """The mem0 Ollama providers need this extra client at first use."""
        class TagsResponse:
            def read(self):
                return b'{"models": [{"name": "mistral:7b"}, {"name": "nomic-embed-text"}]}'

        monkeypatch.setattr(cli.urllib.request, "urlopen", lambda *args, **kwargs: TagsResponse())
        monkeypatch.setitem(sys.modules, "ollama", None)

        with pytest.raises(SystemExit) as exit_info:
            cli.cmd_check(SimpleNamespace())

        assert exit_info.value.code == 1
        assert "Python package 'ollama' is not installed" in capsys.readouterr().out


class TestRuntimePrerequisites:
    def test_query_limit_must_be_positive(self):
        with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
            positive_int("0")

        assert positive_int("3") == 3

    def test_seed_dry_run_does_not_need_runtime_clients(self, tmp_workdir, monkeypatch, capsys):
        source = tmp_workdir / "memory.md"
        source.write_text("# Memory\nA fixture for dry-run output.\n")
        monkeypatch.setitem(sys.modules, "ollama", None)
        monkeypatch.setitem(sys.modules, "mem0", None)

        cli.cmd_seed(SimpleNamespace(source=[str(source)], dry_run=True))

        assert "Would seed 1 chunks" in capsys.readouterr().out

    def test_seed_fails_clearly_without_python_ollama_client(self, tmp_workdir, monkeypatch, capsys):
        source = tmp_workdir / "memory.md"
        source.write_text("# Memory\nA fixture for prerequisite output.\n")
        monkeypatch.setitem(sys.modules, "ollama", None)

        with pytest.raises(SystemExit) as exit_info:
            cli.cmd_seed(SimpleNamespace(source=[str(source)], dry_run=False))

        assert exit_info.value.code == 1
        assert "Python package 'ollama' is not installed" in capsys.readouterr().out

    def test_foreground_serve_propagates_server_failure(self, monkeypatch):
        monkeypatch.setattr(cli, "require_ollama_client", lambda: True)
        monkeypatch.setattr(cli, "load_config", lambda: {})
        monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=7))

        with pytest.raises(SystemExit) as exit_info:
            cli.cmd_serve(SimpleNamespace(port=None, background=False))

        assert exit_info.value.code == 7

    def test_background_serve_waits_for_readiness(self, tmp_workdir, monkeypatch, capsys):
        process = SimpleNamespace(pid=123, poll=lambda: None)
        monkeypatch.setattr(cli, "require_ollama_client", lambda: True)
        monkeypatch.setattr(cli, "load_config", lambda: {})
        monkeypatch.setattr(cli, "port_is_in_use", lambda port: False)
        monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: process)
        monkeypatch.setattr(cli, "wait_for_server", lambda port, process, token: True)

        cli.cmd_serve(SimpleNamespace(port=None, background=True))

        assert "server started (PID 123, port 18420)" in capsys.readouterr().out
        state = json.loads((tmp_workdir / ".zer0dex" / "server.json").read_text())
        assert state["pid"] == 123
        assert state["port"] == 18420
        assert state["token"]

    def test_background_serve_fails_when_not_ready(self, tmp_workdir, monkeypatch):
        calls = []
        process = SimpleNamespace(
            pid=123,
            poll=lambda: None,
            terminate=lambda: calls.append("terminate"),
            wait=lambda timeout=None: calls.append(("wait", timeout)),
        )
        monkeypatch.setattr(cli, "require_ollama_client", lambda: True)
        monkeypatch.setattr(cli, "load_config", lambda: {})
        monkeypatch.setattr(cli, "port_is_in_use", lambda port: False)
        monkeypatch.setattr(cli.subprocess, "Popen", lambda *args, **kwargs: process)
        monkeypatch.setattr(cli, "wait_for_server", lambda port, process, token: False)

        with pytest.raises(SystemExit) as exit_info:
            cli.cmd_serve(SimpleNamespace(port=None, background=True))

        assert exit_info.value.code == 1
        assert calls == ["terminate", ("wait", 5)]
        assert not cli.server_state_file().exists()

    def test_stop_terminates_verified_background_server(self, tmp_workdir, monkeypatch, capsys):
        cli.save_server_state(123, "launch-token", 18420)
        checks = iter([True, False, False])
        signal_calls = []
        monkeypatch.setattr(cli, "process_is_running", lambda pid: next(checks))
        monkeypatch.setattr(cli, "is_managed_server_process", lambda state: True)
        monkeypatch.setattr(cli.os, "kill", lambda pid, sig: signal_calls.append((pid, sig)))

        cli.cmd_stop(SimpleNamespace())

        assert signal_calls == [(123, cli.signal.SIGTERM)]
        assert not (tmp_workdir / ".zer0dex" / "server.json").exists()
        assert "server stopped (PID 123)" in capsys.readouterr().out

    def test_stop_cleans_dead_server_state(self, tmp_workdir, monkeypatch, capsys):
        cli.save_server_state(123, "launch-token", 18420)
        monkeypatch.setattr(cli, "process_is_running", lambda pid: False)

        cli.cmd_stop(SimpleNamespace())

        assert not (tmp_workdir / ".zer0dex" / "server.json").exists()
        assert "Removed stale" in capsys.readouterr().out

    def test_stop_refuses_reused_or_unrelated_pid(self, tmp_workdir, monkeypatch, capsys):
        cli.save_server_state(123, "launch-token", 18420)
        signal_calls = []
        monkeypatch.setattr(cli, "process_is_running", lambda pid: True)
        monkeypatch.setattr(cli, "is_managed_server_process", lambda state: False)
        monkeypatch.setattr(cli.os, "kill", lambda pid, sig: signal_calls.append((pid, sig)))

        with pytest.raises(SystemExit) as exit_info:
            cli.cmd_stop(SimpleNamespace())

        assert exit_info.value.code == 1
        assert signal_calls == []
        assert (tmp_workdir / ".zer0dex" / "server.json").exists()
        assert "refusing to signal the recorded PID" in capsys.readouterr().out

    def test_managed_process_identity_requires_server_and_launch_token(self, monkeypatch):
        state = {"pid": 123, "token": "launch-token", "port": 18420}
        monkeypatch.setattr(cli, "managed_server_pid", lambda port, token: 123)
        assert cli.is_managed_server_process(state)

        monkeypatch.setattr(cli, "managed_server_pid", lambda port, token: 456)
        assert not cli.is_managed_server_process(state)

    def test_server_state_uses_configured_storage_path(self, tmp_workdir):
        config = {"chroma_path": "custom-memory"}

        cli.save_server_state(123, "launch-token", 18420, config)

        state_file = tmp_workdir / "custom-memory" / "server.json"
        assert json.loads(state_file.read_text())["pid"] == 123
        assert state_file.stat().st_mode & 0o777 == 0o600
        assert not (tmp_workdir / ".zer0dex" / "server.json").exists()

    def test_invalid_server_port_is_stale_state(self, tmp_workdir):
        state_file = tmp_workdir / ".zer0dex" / "server.json"
        state_file.parent.mkdir()
        state_file.write_text(
            json.dumps({"pid": 123, "token": "launch-token", "port": 70000})
        )

        assert cli.load_server_state() is None

    def test_boolean_pid_is_stale_state(self, tmp_workdir):
        state_file = tmp_workdir / ".zer0dex" / "server.json"
        state_file.parent.mkdir()
        state_file.write_text(
            json.dumps({"pid": True, "token": "launch-token", "port": 18420})
        )

        assert cli.load_server_state() is None

    def test_stop_is_a_successful_noop_when_repeated(self, tmp_workdir, capsys):
        cli.cmd_stop(SimpleNamespace())

        assert "No managed zer0dex background server is running" in capsys.readouterr().out

    def test_background_serve_rejects_an_occupied_port(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "require_ollama_client", lambda: True)
        monkeypatch.setattr(cli, "load_config", lambda: {})
        monkeypatch.setattr(cli, "port_is_in_use", lambda port: True)

        with pytest.raises(SystemExit) as exit_info:
            cli.cmd_serve(SimpleNamespace(port=18420, background=True))

        assert exit_info.value.code == 1
        assert "port 18420 is already in use" in capsys.readouterr().out

    def test_background_readiness_wait_observes_health_endpoint(self):
        class HealthHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                body = b'{"status": "ok", "count": 0}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", 0), HealthHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        process = SimpleNamespace(poll=lambda: None)
        try:
            assert cli.wait_for_server(server.server_port, process, timeout_seconds=1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class TestCliLocalServerExchange:
    def test_status_query_and_add_against_local_handler(self, capsys):
        """Exercise the CLI HTTP calls without requiring an Ollama runtime."""
        class FakeMemory:
            def __init__(self):
                self.memories = ["Use concise factual replies."]

            def get_all(self, *, filters, top_k):
                assert filters == {"user_id": "agent"}
                return {
                    "results": [
                        {"memory": text} for text in self.memories[:top_k]
                    ]
                }

            def search(self, text, *, filters, top_k):
                assert filters == {"user_id": "agent"}
                return {
                    "results": [
                        {"memory": memory, "score": 0.91}
                        for memory in self.memories[:top_k]
                    ]
                }

            def add(self, text, *, user_id):
                assert user_id == "agent"
                self.memories.append(text)
                return {"results": [{"memory": text}]}

        original_memory = Mem0Handler.memory
        original_user_id = Mem0Handler.user_id
        original_min_score = Mem0Handler.min_score
        Mem0Handler.memory = FakeMemory()
        Mem0Handler.user_id = "agent"
        Mem0Handler.min_score = 0.3
        server = HTTPServer(("127.0.0.1", 0), Mem0Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_port

        try:
            cli.cmd_status(SimpleNamespace(port=port))
            cli.cmd_query(SimpleNamespace(text="reply style", limit=5, port=port))
            cli.cmd_add(SimpleNamespace(text="The service is loopback-only.", port=port))
            cli.cmd_status(SimpleNamespace(port=port))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            Mem0Handler.memory = original_memory
            Mem0Handler.user_id = original_user_id
            Mem0Handler.min_score = original_min_score

        output = capsys.readouterr().out
        assert "Memories: 1" in output
        assert "[0.910] Use concise factual replies." in output
        assert "Added 1 memory(ies)" in output
        assert "Memories: 2" in output
