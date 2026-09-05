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
from http.server import HTTPServer, BaseHTTPRequestHandler

from mem0 import Memory


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
            all_mem = self.memory.get_all(user_id=self.user_id)
            count = len(all_mem.get("results", []))
            self._send_json({"status": "ok", "count": count})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        data = self._read_body()
        if data is None:
            self._send_json({"error": "invalid json"}, 400)
            return
        if not isinstance(data, dict):
            self._send_json({"error": "json body must be an object"}, 400)
            return

        if self.path == "/query":
            text = data.get("text", "")
            limit = data.get("limit", 5)
            min_score = data.get("min_score", self.min_score)
            if not text or len(text.strip()) < 3:
                self._send_json({"memories": []})
                return

            results = self.memory.search(text, user_id=self.user_id, limit=limit)
            memories = []
            for mem in results.get("results", []):
                score = mem.get("score", 0)
                if score > min_score:
                    memories.append(
                        {
                            "text": mem.get("memory", ""),
                            "score": round(score, 3),
                        }
                    )
            self._send_json({"memories": memories})

        elif self.path == "/add":
            text = data.get("text", "")
            if not text:
                self._send_json({"error": "no text"}, 400)
                return
            result = self.memory.add(text, user_id=self.user_id)
            extracted = result.get("results", [])
            self._send_json(
                {
                    "count": len(extracted),
                    "memories": [m.get("memory", "") for m in extracted],
                }
            )

        else:
            self._send_json({"error": "not found"}, 404)


def main():
    parser = argparse.ArgumentParser(description="zer0dex Memory Server")
    parser.add_argument(
        "--port", type=int, default=18420, help="Server port (default: 18420)"
    )
    parser.add_argument(
        "--collection", default="zer0dex", help="ChromaDB collection name"
    )
    parser.add_argument(
        "--chroma-path", default=".zer0dex", help="ChromaDB storage path"
    )
    parser.add_argument("--user-id", default="agent", help="mem0 user ID")
    parser.add_argument(
        "--llm-model", default="mistral:7b", help="Ollama LLM model for extraction"
    )
    parser.add_argument(
        "--embed-model", default="nomic-embed-text", help="Ollama embedding model"
    )
    parser.add_argument(
        "--ollama-url", default="http://localhost:11434", help="Ollama base URL"
    )
    parser.add_argument(
        "--min-score", type=float, default=0.3, help="Minimum relevance score"
    )
    args = parser.parse_args()

    config = build_config(args)
    print(f"Loading mem0 (collection: {args.collection})...", flush=True)
    memory = Memory.from_config(config)
    all_mem = memory.get_all(user_id=args.user_id)
    count = len(all_mem.get("results", []))
    print(f"Ready — {count} memories, serving on port {args.port}", flush=True)

    Mem0Handler.memory = memory
    Mem0Handler.user_id = args.user_id
    Mem0Handler.min_score = args.min_score

    server = HTTPServer(("127.0.0.1", args.port), Mem0Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
