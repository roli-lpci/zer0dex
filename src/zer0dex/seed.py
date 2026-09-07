#!/usr/bin/env python3
"""
Seed mem0 vector store from markdown files.
Reads files, splits into chunks, and feeds to mem0 for extraction + embedding.

Usage:
  python seed.py --source MEMORY.md --source memory/ [--collection my_agent] [--chroma-path ./.mem0_chroma]
"""
import argparse
import sys
from pathlib import Path

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


def collect_files(sources):
    """Collect all .md files from source paths (files or directories)."""
    files = []
    for source in sources:
        p = Path(source)
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.glob("**/*.md")))
        else:
            print(f"Warning: {source} not found, skipping", file=sys.stderr)
    return files


def chunk_markdown(text, max_chunk=2000):
    """Split markdown text into chunks at section boundaries."""
    sections = []
    current = []
    for line in text.split("\n"):
        current_has_content = any(
            existing.strip() and not existing.lstrip().startswith("#")
            for existing in current
        )
        if line.startswith("## ") and current_has_content:
            sections.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current))

    # Further split large sections
    chunks = []
    for section in sections:
        if len(section) <= max_chunk:
            chunks.append(section)
        else:
            words = section.split()
            chunk = []
            length = 0
            for word in words:
                if length + len(word) + 1 > max_chunk and chunk:
                    chunks.append(" ".join(chunk))
                    chunk = [word]
                    length = len(word)
                else:
                    chunk.append(word)
                    length += len(word) + 1
            if chunk:
                chunks.append(" ".join(chunk))
    return [c for c in chunks if c.strip()]


def get_all_for_user(memory, user_id):
    """Read a user's memories across the supported mem0 API shapes."""
    top_k = 100
    try:
        while True:
            result = memory.get_all(
                filters={"user_id": user_id},
                top_k=top_k,
            )
            if len(result.get("results", [])) < top_k:
                return result
            top_k *= 2
    except TypeError:
        # Older mem0 releases accepted entity IDs as top-level parameters.
        return memory.get_all(user_id=user_id)


def search_for_user(memory, text, user_id, limit):
    """Search a user's memories across the supported mem0 API shapes."""
    try:
        return memory.search(text, filters={"user_id": user_id}, top_k=limit)
    except TypeError:
        # Older mem0 releases named the result bound ``limit``.
        return memory.search(text, user_id=user_id, limit=limit)


def main():
    parser = argparse.ArgumentParser(description="Seed zer0dex vector store")
    parser.add_argument("--source", action="append", required=True, help="Source file or directory (can specify multiple)")
    parser.add_argument("--collection", default="zer0dex", help="ChromaDB collection name")
    parser.add_argument("--chroma-path", default=".zer0dex", help="ChromaDB storage path")
    parser.add_argument("--user-id", default="agent", help="mem0 user ID")
    parser.add_argument("--llm-model", default="mistral:7b", help="Ollama LLM model for extraction")
    parser.add_argument("--embed-model", default="nomic-embed-text", help="Ollama embedding model")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be seeded without writing")
    args = parser.parse_args()

    files = collect_files(args.source)
    if not files:
        print("No files found to seed.")
        sys.exit(1)

    print(f"Found {len(files)} file(s) to seed:")
    for f in files:
        print(f"  {f}")

    all_chunks = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        chunks = chunk_markdown(text)
        all_chunks.extend(chunks)
        print(f"  {f.name}: {len(chunks)} chunks")

    print(f"\nTotal: {len(all_chunks)} chunks")

    if args.dry_run:
        print("\n[DRY RUN] Would seed the above chunks. Exiting.")
        sys.exit(0)

    try:
        import ollama  # noqa: F401
    except ImportError:
        print(
            "Error: Python package 'ollama' is not installed. "
            "Install it with: pip install ollama",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from mem0 import Memory
    except ImportError:
        print("Error: mem0ai not installed. Run: pip install mem0ai", file=sys.stderr)
        sys.exit(1)

    config = build_config(args)
    print(f"\nLoading mem0 (collection: {args.collection})...")
    memory = Memory.from_config(config)

    total_extracted = 0
    for i, chunk in enumerate(all_chunks, 1):
        print(f"  Seeding chunk {i}/{len(all_chunks)}...", end=" ", flush=True)
        result = memory.add(chunk, user_id=args.user_id)
        extracted = len(result.get("results", []))
        total_extracted += extracted
        print(f"({extracted} memories extracted)")

    all_mem = get_all_for_user(memory, args.user_id)
    final_count = len(all_mem.get("results", []))
    print(f"\nDone. {total_extracted} memories extracted. Total in store: {final_count}")


if __name__ == "__main__":
    main()
