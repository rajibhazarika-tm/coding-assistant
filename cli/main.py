#!/usr/bin/env python3
# cli/main.py
"""CLI entry point for the local coding assistant."""

import argparse
import sys
from pathlib import Path


def cmd_analyze(args):
    """Analyze a repository and print the recommended indexing strategy."""
    from indexer.strategy import analyze_repo, format_analysis

    analysis = analyze_repo(args.path)
    print("\n" + format_analysis(analysis))


def cmd_index(args):
    """Index a repository."""
    from indexer.strategy import analyze_repo, format_analysis, get_profile

    repo_path = Path(args.path).resolve()
    if not repo_path.exists():
        print(f"Path not found: {repo_path}")
        sys.exit(1)

    if args.auto:
        analysis = analyze_repo(repo_path)
        print("\n" + format_analysis(analysis) + "\n")
        if args.dry_run:
            print("Dry run complete. No cache, chunks, or embeddings were written.")
            return

        total_indexed = 0
        for area in analysis.areas:
            total_indexed += _index_single_path(area.path, area.profile, args.force, dry_run=False)
        print(f"\nAuto indexing complete. Indexed {total_indexed} chunks across {len(analysis.areas)} area(s).")
        _print_stats()
        return

    profile = get_profile(args.profile)
    _index_single_path(repo_path, profile.name, args.force, dry_run=args.dry_run)


def _index_single_path(repo_path: Path, profile_name: str, force: bool, dry_run: bool = False) -> int:
    from indexer.scanner import scan_repo, repo_id_for_path
    from indexer.chunker import chunk_file
    from indexer.embedder import index_chunks, delete_chunks_for_files, delete_chunks_for_repo
    from indexer.strategy import get_profile

    profile = get_profile(profile_name)
    print(f"\nIndexing: {repo_path}")
    print(f"   Mode: {'full re-index' if force else 'incremental'}")
    print(f"   Profile: {profile.name} - {profile.description}\n")

    repo_id = repo_id_for_path(repo_path)
    files, deleted = scan_repo(
        repo_path,
        incremental=True,
        force_reindex=force,
        include_extensions=profile.include_extensions,
        include_filenames=profile.include_filenames,
        extra_skip_dirs=profile.extra_skip_dirs,
        max_file_size_kb=profile.max_file_size_kb,
        update_cache=not dry_run,
    )

    if dry_run:
        print(f"Dry run: {len(files):,} files would be indexed; {len(deleted):,} deleted files detected.")
        print("No cache, chunks, or embeddings were written.")
        return 0

    if force:
        removed = delete_chunks_for_repo(repo_id)
        if removed:
            print(f"Removed {removed} existing chunks for this repo")

    if deleted:
        removed = delete_chunks_for_files(deleted, repo_id=repo_id)
        print(f"Removed {removed} chunks for {len(deleted)} deleted files")

    if not files:
        print("Nothing to index; all files are up to date.")
        _print_stats()
        return 0

    print(f"\nChunking {len(files)} files...")
    all_chunks = []
    for scanned_file in files:
        try:
            content = scanned_file.path.read_text(encoding="utf-8", errors="replace")
            chunks = chunk_file(
                scanned_file.relative_path,
                scanned_file.language,
                content,
                repo_id=scanned_file.repo_id,
            )
            all_chunks.extend(chunks)
        except Exception as exc:
            print(f"   {scanned_file.relative_path}: {exc}")

    print(f"   -> {len(all_chunks)} semantic chunks extracted")

    changed_paths = sorted({scanned_file.relative_path for scanned_file in files})
    removed = delete_chunks_for_files(changed_paths, repo_id=repo_id)
    if removed:
        print(f"Removed {removed} stale chunks for changed files")

    print("\nEmbedding and storing in ChromaDB...")
    indexed = index_chunks(all_chunks, show_progress=True)

    print(f"\nDone. Indexed {indexed} chunks from {len(files)} files.")
    _print_stats()
    return indexed


def cmd_ask(args):
    """Answer a single question with RAG."""
    from retriever.hybrid_search import retrieve
    from retriever.context_builder import build_context
    from assistant.prompts import build_prompt, build_no_context_prompt
    from assistant.llm import ask

    _check_ollama()

    query = args.question
    print("\nRetrieving relevant context...", end="", flush=True)

    chunks = retrieve(query, top_k=args.top_k or 5)
    if chunks:
        print(f" found {len(chunks)} chunks\n")
        context, sources = build_context(chunks, query, task_type="general")
        system, user_msg = build_prompt("general", query, context, sources)
    else:
        print(" no index found, using model knowledge only\n")
        system, user_msg = build_no_context_prompt("general", query)
        sources = []

    if sources:
        print(f"Sources: {', '.join(sources[:3])}\n")

    print("-" * 60)
    ask(system, user_msg, print_streaming=True)
    print("-" * 60)


def cmd_review(args):
    """Review a specific file."""
    from retriever.hybrid_search import retrieve
    from retriever.context_builder import build_context
    from assistant.prompts import build_prompt
    from assistant.llm import ask

    _check_ollama()

    file_path = args.file
    query = f"Review this file for bugs, security issues, code quality, and improvements: {file_path}"

    print(f"\nLoading context for {file_path}...")
    chunks = retrieve(query, file_filter=file_path, top_k=8)

    if not chunks:
        try:
            content = Path(file_path).read_text(encoding="utf-8", errors="replace")
            if len(content) > 8000:
                content = content[:8000] + "\n# [... file truncated ...]"
            context = f"### {file_path}\n```\n{content}\n```"
            sources = [file_path]
        except Exception as exc:
            print(f"Cannot read file: {exc}")
            return
    else:
        context, sources = build_context(chunks, query, "review")

    system, user_msg = build_prompt("review", query, context, sources)
    print("\nCode Review\n" + "-" * 60)
    ask(system, user_msg, print_streaming=True)
    print("-" * 60)


def cmd_explain(args):
    """Explain a function or file."""
    from retriever.hybrid_search import retrieve
    from retriever.context_builder import build_context
    from assistant.prompts import build_prompt
    from assistant.llm import ask

    _check_ollama()

    if args.function:
        query = f"Explain the function `{args.function}` in {args.file or 'the codebase'}"
    elif args.file:
        query = f"Explain what {args.file} does and how it works"
    else:
        print("Provide --file and/or --function")
        return

    chunks = retrieve(query, file_filter=args.file, top_k=5)
    context, sources = build_context(chunks, query, "explain") if chunks else ("", [])
    system, user_msg = build_prompt("explain", query, context, sources)

    print("\nExplanation\n" + "-" * 60)
    ask(system, user_msg, print_streaming=True)
    print("-" * 60)


def cmd_generate(args):
    """Generate code from a description."""
    from retriever.hybrid_search import retrieve
    from retriever.context_builder import build_context
    from assistant.prompts import build_prompt, build_no_context_prompt
    from assistant.llm import ask

    _check_ollama()

    query = args.description
    print("\nFinding similar code patterns...")

    chunks = retrieve(query, top_k=4)
    if chunks:
        context, sources = build_context(chunks, query, "generate")
        extra = "Match the coding style, patterns, and conventions from the context above."
        system, user_msg = build_prompt("generate", query, context, sources, extra_instruction=extra)
    else:
        system, user_msg = build_no_context_prompt("generate", query)

    print("\nGenerating code...\n" + "-" * 60)
    ask(system, user_msg, print_streaming=True)
    print("-" * 60)


def cmd_chat(args):
    """Interactive multi-turn chat."""
    from retriever.hybrid_search import retrieve
    from retriever.context_builder import build_context
    from assistant.llm import ask

    _check_ollama()

    print("\nCoding Assistant (Interactive Mode)")
    print("Type your question. Commands: /exit, /clear, /help\n")

    history = []
    system = "You are a helpful coding assistant. Answer questions about the codebase using the provided context. Be concise and accurate."

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
            print("Bye!")
            break
        if user_input.lower() == "/clear":
            history = []
            print("History cleared\n")
            continue
        if user_input.lower() == "/help":
            print("\nCommands:\n  /clear  - Clear conversation history\n  /exit   - Quit\n")
            continue

        chunks = retrieve(user_input, top_k=4)
        if chunks:
            context, _ = build_context(chunks, user_input)
            context_msg = f"{context}\n\n---\n\n{user_input}"
        else:
            context_msg = user_input

        print("\nAssistant: ", end="", flush=True)
        response = ask(system, context_msg, history=history, print_streaming=True)
        print()

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        if len(history) > 12:
            history = history[-12:]


def cmd_stats(args):
    """Show index statistics."""
    from indexer.embedder import get_collection_stats

    stats = get_collection_stats()
    print("\nIndex Statistics")
    print(f"   Total chunks: {stats['total_chunks']:,}")
    print(f"   Status: {stats['status']}")


def _check_ollama():
    """Exit with a helpful message if Ollama isn't running."""
    from assistant.llm import check_ollama_running, check_model_available
    from config.settings import MODEL

    if not check_ollama_running():
        print("Ollama is not running. Start it with: ollama serve")
        sys.exit(1)

    if not check_model_available(MODEL):
        print(f"Model '{MODEL}' not found. Pull it with: ollama pull {MODEL}")
        sys.exit(1)


def _print_stats():
    from indexer.embedder import get_collection_stats

    stats = get_collection_stats()
    print(f"\nIndex now has {stats['total_chunks']:,} total chunks")


def main():
    parser = argparse.ArgumentParser(
        description="Local Coding Assistant (qwen2.5-coder + Ollama)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m cli.main analyze --path ~/projects/myapp
  python -m cli.main index --path ~/projects/myapp --auto --dry-run
  python -m cli.main index --path ~/projects/myapp --auto
  python -m cli.main ask "How does the auth middleware work?"
  python -m cli.main review --file src/auth.py
  python -m cli.main chat
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_analyze = subparsers.add_parser("analyze", help="Analyze a repo and recommend an index strategy")
    p_analyze.add_argument("--path", required=True, help="Path to repository root")

    p_index = subparsers.add_parser("index", help="Scan and index a repository")
    p_index.add_argument("--path", required=True, help="Path to repository root")
    p_index.add_argument("--force", action="store_true", help="Force full re-index")
    p_index.add_argument("--auto", action="store_true", help="Analyze repo and choose indexing profiles automatically")
    p_index.add_argument("--dry-run", action="store_true", help="Show what would be indexed without writing anything")
    p_index.add_argument(
        "--profile",
        default="generic",
        choices=["generic", "spring", "java-library", "frontend"],
        help="Manual indexing profile when --auto is not used",
    )

    p_ask = subparsers.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", help="Your question")
    p_ask.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve")

    subparsers.add_parser("chat", help="Interactive multi-turn chat")

    p_review = subparsers.add_parser("review", help="Review a file")
    p_review.add_argument("--file", required=True, help="File path to review")

    p_explain = subparsers.add_parser("explain", help="Explain code")
    p_explain.add_argument("--file", help="File to explain")
    p_explain.add_argument("--function", help="Function name to explain")

    p_gen = subparsers.add_parser("generate", help="Generate code")
    p_gen.add_argument("description", help="What to generate")

    subparsers.add_parser("stats", help="Show index statistics")

    args = parser.parse_args()
    commands = {
        "analyze": cmd_analyze,
        "index": cmd_index,
        "ask": cmd_ask,
        "chat": cmd_chat,
        "review": cmd_review,
        "explain": cmd_explain,
        "generate": cmd_generate,
        "stats": cmd_stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
