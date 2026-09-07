#!/usr/bin/env python3
"""
zer0dex CLI — Dual-layer memory for AI agents.

Commands:
  zer0dex init     Initialize a new zer0dex memory store
  zer0dex seed     Seed vector store from markdown files
  zer0dex serve    Start the memory server
  zer0dex stop     Stop this project's managed background server
  zer0dex query    Query memories from the command line
  zer0dex status   Check server health and memory count
  zer0dex add      Add a memory manually
"""
import argparse
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path


DEFAULT_PORT = 18420
DEFAULT_COLLECTION = "zer0dex"
DEFAULT_CHROMA_PATH = ".zer0dex"
DEFAULT_LLM_MODEL = "mistral:7b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_USER_ID = "agent"

CONFIG_FILE = ".zer0dex.json"


def positive_int(value):
    """Parse a strictly positive integer for CLI result limits."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def require_ollama_client():
    """Return whether the Python Ollama client required by mem0 is installed."""
    try:
        import ollama  # noqa: F401
    except ImportError:
        print(
            "Error: Python package 'ollama' is not installed. "
            "Install it with: pip install ollama"
        )
        return False
    return True


def port_is_in_use(port):
    """Return whether another local process is already listening on the port."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def wait_for_server(port, process, token=None, timeout_seconds=30):
    """Wait for the spawned server to answer health and identity checks."""
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            print(f"Error: zer0dex server exited before readiness (exit {process.returncode}).")
            return False
        try:
            response = urllib.request.urlopen(url, timeout=1)
            payload = json.loads(response.read())
            if payload.get("status") == "ok" and (
                token is None or managed_server_pid(port, token) == process.pid
            ):
                return True
        except (OSError, json.JSONDecodeError):
            pass
        time.sleep(0.2)

    print(f"Error: zer0dex server did not become ready within {timeout_seconds} seconds.")
    return False


def server_state_file(config=None):
    """Return the lifecycle state path for this project's configured store."""
    config = load_config() if config is None else config
    return Path(config.get("chroma_path", DEFAULT_CHROMA_PATH)) / "server.json"


def load_server_state(config=None):
    """Return the background-server state, or None when it is absent or invalid."""
    state_file = server_state_file(config)
    try:
        state = json.loads(state_file.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    if not isinstance(state, dict):
        return None
    if (
        not isinstance(state.get("pid"), int)
        or isinstance(state["pid"], bool)
        or state["pid"] < 1
    ):
        return None
    if not isinstance(state.get("token"), str) or not state["token"]:
        return None
    if (
        not isinstance(state.get("port"), int)
        or isinstance(state["port"], bool)
        or not 1 <= state["port"] <= 65535
    ):
        return None
    return state


def save_server_state(pid, token, port, config=None):
    """Record the exact background child that this project started."""
    state_file = server_state_file(config)
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=state_file.parent,
            prefix=f".{state_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(
                json.dumps({"pid": pid, "token": token, "port": port}) + "\n"
            )
            temporary = Path(handle.name)
        temporary.replace(state_file)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def clear_server_state(config=None):
    """Remove a stale local server record without touching any process."""
    try:
        server_state_file(config).unlink()
    except FileNotFoundError:
        pass


def managed_server_pid(port, token):
    """Return the PID from a background server that proves the launch token."""
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/_lifecycle",
        headers={"X-Zer0dex-Instance-Token": token},
    )
    try:
        response = urllib.request.urlopen(request, timeout=1)
        payload = json.loads(response.read())
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return None
    pid = payload.get("pid") if isinstance(payload, dict) else None
    return pid if isinstance(pid, int) and pid > 0 else None


def is_managed_server_process(state):
    """Confirm PID ownership through the server's per-launch token handshake."""
    return managed_server_pid(state["port"], state["token"]) == state["pid"]


def process_is_running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_spawned_process(process):
    """Stop and reap a child that failed background startup."""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def load_config():
    """Load config from .zer0dex.json if it exists."""
    p = Path(CONFIG_FILE)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_config(config):
    """Save config to .zer0dex.json."""
    Path(CONFIG_FILE).write_text(json.dumps(config, indent=2) + "\n")


def cmd_init(args):
    """Initialize a new zer0dex memory store."""
    config = {
        "collection": args.collection or DEFAULT_COLLECTION,
        "chroma_path": args.chroma_path or DEFAULT_CHROMA_PATH,
        "port": args.port or DEFAULT_PORT,
        "user_id": args.user_id or DEFAULT_USER_ID,
        "llm_model": DEFAULT_LLM_MODEL,
        "embed_model": DEFAULT_EMBED_MODEL,
        "ollama_url": DEFAULT_OLLAMA_URL,
    }
    save_config(config)
    Path(config["chroma_path"]).mkdir(parents=True, exist_ok=True)

    print("✅ zer0dex initialized")
    print(f"   Collection: {config['collection']}")
    print(f"   Storage: {config['chroma_path']}")
    print(f"   Config: {CONFIG_FILE}")
    print("\nNext: zer0dex seed --source your-docs/")


def cmd_seed(args):
    """Seed the vector store from files."""
    config = load_config()
    # Import here to avoid slow startup for other commands
    from zer0dex.seed import collect_files, chunk_markdown, get_all_for_user

    files = collect_files(args.source)
    if not files:
        print("No files found. Use --source <path>")
        sys.exit(1)

    print(f"Found {len(files)} file(s)")
    all_chunks = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks = chunk_markdown(text)
        all_chunks.extend(chunks)
        print(f"  {f.name}: {len(chunks)} chunks")

    if args.dry_run:
        print(f"\n[DRY RUN] Would seed {len(all_chunks)} chunks. Exiting.")
        return

    if not require_ollama_client():
        sys.exit(1)

    try:
        from mem0 import Memory
    except ImportError:
        print("Error: mem0ai not installed. Run: pip install mem0ai")
        sys.exit(1)

    mem_config = {
        "llm": {"provider": "ollama", "config": {"model": config.get("llm_model", DEFAULT_LLM_MODEL), "ollama_base_url": config.get("ollama_url", DEFAULT_OLLAMA_URL)}},
        "embedder": {"provider": "ollama", "config": {"model": config.get("embed_model", DEFAULT_EMBED_MODEL), "ollama_base_url": config.get("ollama_url", DEFAULT_OLLAMA_URL)}},
        "vector_store": {"provider": "chroma", "config": {"collection_name": config.get("collection", DEFAULT_COLLECTION), "path": config.get("chroma_path", DEFAULT_CHROMA_PATH)}},
    }

    print("\nLoading mem0...")
    memory = Memory.from_config(mem_config)
    user_id = config.get("user_id", DEFAULT_USER_ID)

    total = 0
    for i, chunk in enumerate(all_chunks, 1):
        print(f"  Seeding {i}/{len(all_chunks)}...", end=" ", flush=True)
        result = memory.add(chunk, user_id=user_id)
        n = len(result.get("results", []))
        total += n
        print(f"({n} memories)")

    all_mem = get_all_for_user(memory, user_id)
    final = len(all_mem.get("results", []))
    print(f"\n✅ Seeded {total} memories. Total in store: {final}")


def cmd_check(args):
    """Validate prerequisites before init or seed."""
    config = load_config()
    ollama_url = config.get("ollama_url", DEFAULT_OLLAMA_URL)
    llm_model = config.get("llm_model", DEFAULT_LLM_MODEL)
    embed_model = config.get("embed_model", DEFAULT_EMBED_MODEL)
    all_ok = True

    # 1. Ollama running
    try:
        resp = urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5)
        tags_data = json.loads(resp.read())
        print(f"✅ Ollama is running at {ollama_url}")

        # 2. Required models present
        available = [m.get("name", "") for m in tags_data.get("models", [])]
        for model in [embed_model, llm_model]:
            # Match by prefix (e.g. "mistral:7b" matches "mistral:7b-instruct")
            found = any(m == model or m.startswith(model.split(":")[0] + ":") for m in available)
            if found:
                print(f"✅ Model present: {model}")
            else:
                print(f"❌ Model missing: {model}  (run: ollama pull {model})")
                all_ok = False

    except urllib.error.URLError:
        print(f"❌ Ollama not reachable at {ollama_url}  (run: ollama serve)")
        print(f"❌ Model check skipped: {embed_model}")
        print(f"❌ Model check skipped: {llm_model}")
        all_ok = False

    # 3. mem0ai importable
    try:
        import mem0  # noqa: F401
        print("✅ mem0ai is importable")
    except ImportError:
        print("❌ mem0ai not installed  (run: pip install mem0ai)")
        all_ok = False

    # 4. chromadb importable
    try:
        import chromadb  # noqa: F401
        print("✅ chromadb is importable")
    except ImportError:
        print("❌ chromadb not installed  (run: pip install chromadb)")
        all_ok = False

    # 5. mem0's Ollama providers require the separate Python client.
    if require_ollama_client():
        print("✅ Python Ollama client is importable")
    else:
        all_ok = False

    if not all_ok:
        sys.exit(1)


def cmd_serve(args):
    """Start the memory server."""
    if not require_ollama_client():
        sys.exit(1)

    config = load_config()
    port = args.port or config.get("port", DEFAULT_PORT)

    # Build server args
    server_args = [
        sys.executable, "-m", "zer0dex.server",
        "--port", str(port),
        "--collection", config.get("collection", DEFAULT_COLLECTION),
        "--chroma-path", config.get("chroma_path", DEFAULT_CHROMA_PATH),
        "--user-id", config.get("user_id", DEFAULT_USER_ID),
        "--llm-model", config.get("llm_model", DEFAULT_LLM_MODEL),
        "--embed-model", config.get("embed_model", DEFAULT_EMBED_MODEL),
        "--ollama-url", config.get("ollama_url", DEFAULT_OLLAMA_URL),
    ]

    if args.background:
        state_file = server_state_file(config)
        state = load_server_state(config)
        if state is not None:
            if process_is_running(state["pid"]):
                if is_managed_server_process(state):
                    print(
                        "Error: a managed zer0dex background server is already "
                        "running; use zer0dex stop first."
                    )
                else:
                    print(
                        "Error: the recorded background PID is running but its "
                        "identity cannot be verified; refusing to replace its state."
                    )
                sys.exit(1)
            clear_server_state(config)
        elif state_file.exists():
            clear_server_state(config)
        if port_is_in_use(port):
            print(f"Error: port {port} is already in use; no server was started.")
            sys.exit(1)
        token = secrets.token_urlsafe(24)
        proc = subprocess.Popen(
            server_args + ["--instance-token", token],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            save_server_state(proc.pid, token, port, config)
        except OSError as exc:
            terminate_spawned_process(proc)
            print(f"Error: could not record background server state: {exc}")
            sys.exit(1)
        try:
            ready = wait_for_server(port, proc, token)
        except BaseException:
            terminate_spawned_process(proc)
            clear_server_state(config)
            raise
        if not ready:
            terminate_spawned_process(proc)
            clear_server_state(config)
            sys.exit(1)
        print(f"✅ zer0dex server started (PID {proc.pid}, port {port})")
    else:
        result = subprocess.run(server_args)
        if result.returncode:
            sys.exit(result.returncode)


def cmd_stop(args):
    """Stop the background server started by this project, if it still matches state."""
    config = load_config()
    state_file = server_state_file(config)
    state = load_server_state(config)
    if state is None:
        if state_file.exists():
            clear_server_state(config)
            print("Removed invalid stale zer0dex background server state.")
            return
        print("No managed zer0dex background server is running.")
        return

    pid = state["pid"]
    if not process_is_running(pid):
        clear_server_state(config)
        print("Removed stale zer0dex background server state.")
        return

    if not is_managed_server_process(state):
        print(
            "Error: refusing to signal the recorded PID because the server identity "
            "could not be verified; state was kept for inspection."
        )
        sys.exit(1)

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        clear_server_state(config)
        print("Removed stale zer0dex background server state.")
        return
    except PermissionError:
        print(f"Error: permission denied while stopping zer0dex server PID {pid}; state was kept.")
        sys.exit(1)
    except OSError as exc:
        print(f"Error: could not stop zer0dex server PID {pid}: {exc}; state was kept.")
        sys.exit(1)
    deadline = time.monotonic() + 5
    while process_is_running(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if process_is_running(pid):
        print(f"Error: zer0dex server PID {pid} did not stop; state was kept.")
        sys.exit(1)

    clear_server_state(config)
    print(f"✅ zer0dex server stopped (PID {pid}).")


def cmd_query(args):
    """Query the running server."""
    config = load_config()
    port = args.port or config.get("port", DEFAULT_PORT)
    url = f"http://127.0.0.1:{port}/query"
    data = json.dumps({"text": args.text, "limit": args.limit}).encode()

    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=5)
        result = json.loads(resp.read())
        memories = result.get("memories", [])
        if not memories:
            print("No relevant memories found.")
            return
        for m in memories:
            score = m.get("score", 0)
            text = m.get("text", "")
            print(f"  [{score:.3f}] {text}")
    except urllib.error.URLError:
        print(f"Error: zer0dex server not running on port {port}. Run: zer0dex serve")
        sys.exit(1)


def cmd_status(args):
    """Check server health."""
    config = load_config()
    port = args.port or config.get("port", DEFAULT_PORT)
    url = f"http://127.0.0.1:{port}/health"

    try:
        resp = urllib.request.urlopen(url, timeout=3)
        result = json.loads(resp.read())
        print(f"✅ zer0dex running on port {port}")
        print(f"   Memories: {result.get('count', '?')}")
        print(f"   Status: {result.get('status', '?')}")
    except urllib.error.URLError:
        print(f"❌ zer0dex not running on port {port}")
        sys.exit(1)


def cmd_add(args):
    """Add a memory via the server."""
    config = load_config()
    port = args.port or config.get("port", DEFAULT_PORT)
    url = f"http://127.0.0.1:{port}/add"
    data = json.dumps({"text": args.text}).encode()

    try:
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
        count = result.get("count", 0)
        memories = result.get("memories", [])
        print(f"✅ Added {count} memory(ies):")
        for m in memories:
            print(f"  • {m}")
    except urllib.error.URLError:
        print(f"Error: zer0dex server not running on port {port}. Run: zer0dex serve")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="zer0dex",
        description="Local dual-layer memory for AI agents.",
    )
    sub = parser.add_subparsers(dest="command")

    # check
    sub.add_parser("check", help="Validate prerequisites (Ollama, models, mem0ai, chromadb)")

    # init
    p_init = sub.add_parser("init", help="Initialize a new memory store")
    p_init.add_argument("--collection", help="Collection name")
    p_init.add_argument("--chroma-path", help="ChromaDB storage path")
    p_init.add_argument("--port", type=int, help="Server port")
    p_init.add_argument("--user-id", help="User ID for memories")

    # seed
    p_seed = sub.add_parser("seed", help="Seed from markdown files")
    p_seed.add_argument("--source", action="append", required=True, help="Source file or directory")
    p_seed.add_argument("--dry-run", action="store_true", help="Show what would be seeded")
    p_seed.add_argument("--port", type=int)

    # serve
    p_serve = sub.add_parser("serve", help="Start memory server")
    p_serve.add_argument("--port", type=int, help="Server port")
    p_serve.add_argument("--background", "-b", action="store_true", help="Run in background")

    # stop
    sub.add_parser("stop", help="Stop the project-managed background server")

    # query
    p_query = sub.add_parser("query", help="Query memories")
    p_query.add_argument("text", help="Query text")
    p_query.add_argument("--limit", type=positive_int, default=5, help="Max results")
    p_query.add_argument("--port", type=int)

    # status
    p_status = sub.add_parser("status", help="Check server health")
    p_status.add_argument("--port", type=int)

    # add
    p_add = sub.add_parser("add", help="Add a memory")
    p_add.add_argument("text", help="Memory text")
    p_add.add_argument("--port", type=int)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "check": cmd_check,
        "init": cmd_init,
        "seed": cmd_seed,
        "serve": cmd_serve,
        "stop": cmd_stop,
        "query": cmd_query,
        "status": cmd_status,
        "add": cmd_add,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
