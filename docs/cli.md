# CLI contract (developer preview)

`zer0dex` stores memories locally by default, and its server listens on
`127.0.0.1` only.

Run commands from the directory that should own the store. `zer0dex init`
creates `.zer0dex.json` and, by default, the `.zer0dex/` Chroma directory in
that current directory. The configured `chroma_path` is interpreted by Chroma;
keep it relative to the same project directory or use an explicit absolute path
when moving a store.

## Prerequisites

Use Python 3.11 or 3.12 for the CI-tested developer-preview path. Install the
package, then ensure the local Ollama service and both default models exist:

```bash
pip install -e ".[dev]"
ollama serve
ollama pull nomic-embed-text
ollama pull mistral:7b
zer0dex check
```

`check` tests the service, configured models, `mem0ai`, `chromadb`, and the
Python `ollama` client. It exits 0 only when all checks pass; it exits 1 with
one or more explicit failed or skipped lines otherwise. It does not modify a
store.

## Commands

```bash
zer0dex init [--collection NAME] [--chroma-path PATH] [--port PORT] [--user-id ID]
zer0dex seed --source FILE_OR_DIRECTORY [--source MORE] [--dry-run]
zer0dex serve [--port PORT] [--background]
zer0dex stop
zer0dex status [--port PORT]
zer0dex query TEXT [--limit N] [--port PORT]
zer0dex add TEXT [--port PORT]
```

- `init` writes configuration and creates its storage directory. It exits 0 on
  success. It intentionally does not start Ollama; run `check` before the
  first real seed or server start.
- `seed` recursively reads only `*.md` files from source directories and adds
  extracted memories under the configured user ID. `--dry-run` lists chunks
  and exits 0 without requiring `mem0ai`, the Python Ollama client, or a
  running Ollama service. A missing source or first-use prerequisite exits 1.
- `serve` loads the configured store and starts the loopback HTTP server.
  Foreground mode returns its server exit code. `--background` waits up to 30
  seconds for `/health`; it prints success only after that check passes, and
  exits 1 (terminating and reaping the child) when readiness is not reached. A
  background start writes `server.json` in the configured storage directory
  with a per-launch identity token, and reports success only after the new
  server proves both its token and PID.
- `stop` stops the background server recorded by this project. Before sending
  `SIGTERM`, it asks the loopback server on the recorded port to prove the
  per-launch token and PID. Missing or dead state is removed safely; an
  unverifiable, mismatched, or reused PID is never signaled and causes exit 1
  with state preserved for inspection. Repeating `stop` after a successful
  stop is a successful no-op.
- `status`, `query`, and `add` require a running local server. They exit 1
  when it cannot be reached. `query` exits 0 with `No relevant memories found.`
  when the server returns an empty result set. `query --limit` requires a
  positive integer; invalid values are rejected by the CLI before any request.

The defaults are collection `zer0dex`, user ID `agent`, port `18420`, storage
`.zer0dex`, Ollama URL `http://localhost:11434`, embedding model
`nomic-embed-text`, and extraction model `mistral:7b`. Change configuration
deliberately after `init`; command-line options presently cover collection,
storage, port, and user ID as shown above.

## Disposable workflow

```bash
mkdir demo-memory && cd demo-memory
printf '# Notes\n\n## Preference\nUse concise factual replies.\n' > notes.md
zer0dex check
zer0dex init --port 18429
zer0dex seed --source notes.md
zer0dex serve --port 18429 --background
zer0dex status --port 18429
zer0dex query 'What reply style is preferred?' --port 18429
zer0dex add 'The demo service is local-only.' --port 18429
zer0dex stop
```

This is a local implementation pattern, not a guarantee of complete recall or
correct retrieval for another workload. See [HTTP contract](http.md) and
[compatibility](compatibility.md).
