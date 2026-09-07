#!/usr/bin/env python3
"""
zer0dex Memory Server
Keeps mem0 loaded in memory, responds to queries in ~70ms avg.

Endpoints:
  POST /query  {"text": "...", "limit": 5}  → {"memories": [...]}
  POST /add    {"text": "..."}              → {"count": N}
  GET  /health                               → {"status": "ok", "count": N}

Usage:
  python server.py [--port 18420] [--collection my_agent] [--chroma-path ./.mem0_chroma]
"""
import argparse
import json
from numbers import Real
import os
import secrets
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    from mem0 import Memory
except ImportError:
    Memory = None

from zer0dex.seed import get_all_for_user, search_for_user


def build_config(args):
    return {
        "llm": {
            "provider": "ollama",
            "config": {
                "model": args.llm_model,
                "ollama_base_url": args.ollama_url,
            },
        },
        "embedder": {
            "provider": "ollama",
            "config": {
                "model": args.embed_model,
                "ollama_base_url": args.ollama_url,
            },
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": args.collection,
                "path": args.chroma_path,
            },
        },
    }


class Mem0Handler(BaseHTTPRequestHandler):
    memory = None
    user_id = None
    min_score = 0.3
    instance_token = None

    def log_message(self, format, *args):
        pass  # suppress request logging

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        content_length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return None

    def do_GET(self):
        if self.path == "/health":
            all_mem = get_all_for_user(self.memory, self.user_id)
            count = len(all_mem.get("results", []))
            self._send_json({"status": "ok", "count": count})
        elif self.path == "/_lifecycle":
            token = self.headers.get("X-Zer0dex-Instance-Token", "")
            if not self.instance_token or not secrets.compare_digest(
                token, self.instance_token
            ):
                self._send_json({"error": "not found"}, 404)
                return
            self._send_json({"pid": os.getpid()})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        data = self._read_body()
        if not isinstance(data, dict):
            self._send_json({"error": "invalid json"}, 400)
            return

        if self.path == "/query":
            text = data.get("text", "")
            limit = data.get("limit", 5)
            min_score = data.get("min_score", self.min_score)
            if not isinstance(text, str):
                self._send_json({"error": "text must be a string"}, 400)
                return
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                self._send_json({"error": "limit must be a positive integer"}, 400)
                return
            if not isinstance(min_score, Real) or isinstance(min_score, bool):
                self._send_json({"error": "min_score must be a number"}, 400)
                return
            if len(text.strip()) < 3:
                self._send_json({"memories": []})
                return

            results = search_for_user(self.memory, text, self.user_id, limit)
            memories = []
            for mem in results.get("results", []):
                score = mem.get("score", 0)
                if isinstance(score, Real) and score > min_score:
                    memories.append({
                        "text": mem.get("memory", ""),
                        "score": round(score, 3),
                    })
            self._send_json({"memories": memories})

        elif self.path == "/add":
            text = data.get("text", "")
            if not isinstance(text, str) or not text.strip():
                self._send_json({"error": "text must be a non-empty string"}, 400)
                return
            result = self.memory.add(text, user_id=self.user_id)
            extracted = result.get("results", [])
            self._send_json({
                "count": len(extracted),
                "memories": [m.get("memory", "") for m in extracted],
            })

        else:
            self._send_json({"error": "not found"}, 404)


def main():
    parser = argparse.ArgumentParser(description="zer0dex Memory Server")
    parser.add_argument("--port", type=int, default=18420, help="Server port (default: 18420)")
    parser.add_argument("--collection", default="zer0dex", help="ChromaDB collection name")
    parser.add_argument("--chroma-path", default=".zer0dex", help="ChromaDB storage path")
    parser.add_argument("--user-id", default="agent", help="mem0 user ID")
    parser.add_argument("--llm-model", default="mistral:7b", help="Ollama LLM model for extraction")
    parser.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum relevance score")
    parser.add_argument("--instance-token", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if Memory is None:
        print("Error: mem0ai not installed. Run: pip install mem0ai", file=sys.stderr)
        return 1
    try:
        import ollama  # noqa: F401
    except ImportError:
        print(
            "Error: Python package 'ollama' is not installed. "
            "Install it with: pip install ollama",
            file=sys.stderr,
        )
        return 1

    config = build_config(args)
    print(f"Loading mem0 (collection: {args.collection})...", flush=True)
    try:
        memory = Memory.from_config(config)
        all_mem = get_all_for_user(memory, args.user_id)
    except Exception as exc:
        print(f"Error: could not start local memory runtime: {exc}", file=sys.stderr)
        print("Run `zer0dex check` to verify the Ollama service and models.", file=sys.stderr)
        return 1
    count = len(all_mem.get("results", []))
    print(f"Ready — {count} memories, serving on port {args.port}", flush=True)

    Mem0Handler.memory = memory
    Mem0Handler.user_id = args.user_id
    Mem0Handler.min_score = args.min_score
    Mem0Handler.instance_token = args.instance_token

    server = HTTPServer(("127.0.0.1", args.port), Mem0Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
