# indexer/scanner.py
"""
Scans a repository efficiently:
- Respects .gitignore rules
- Skips binary/generated files
- Tracks file hashes for incremental re-indexing
- Returns only changed files to avoid redundant embedding work
"""

import os
import json
import hashlib
import fnmatch
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass

from config.settings import (
    SUPPORTED_EXTENSIONS, SKIP_DIRS, SKIP_FILES,
    MAX_FILE_SIZE_KB, INDEX_DIR
)


@dataclass
class ScannedFile:
    path: Path
    language: str
    size_bytes: int
    relative_path: str  # relative to repo root
    repo_id: str


def repo_id_for_path(repo_root: str | Path) -> str:
    """Return a stable, filesystem-safe ID for a repository root."""
    root = str(Path(repo_root).resolve()).lower()
    return hashlib.sha1(root.encode("utf-8")).hexdigest()[:12]


def _hash_cache_file(repo_id: str) -> Path:
    return INDEX_DIR / f"file_hashes_{repo_id}.json"


def _load_gitignore_patterns(repo_root: Path) -> list[str]:
    """Load patterns from .gitignore file at repo root."""
    patterns = []
    gitignore = repo_root / ".gitignore"
    if gitignore.exists():
        for line in gitignore.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def _is_gitignored(path: Path, repo_root: Path, patterns: list[str]) -> bool:
    """Check if a path matches any .gitignore pattern."""
    rel = path.relative_to(repo_root)
    rel_str = rel.as_posix()
    rel_parts = rel.parts
    ignored = False

    for pattern in patterns:
        negated = pattern.startswith("!")
        if negated:
            pattern = pattern[1:]
        anchored = pattern.startswith("/")
        pattern = pattern.lstrip("/")
        if not pattern:
            continue
        matched = False
        # Match against full relative path
        if fnmatch.fnmatch(rel_str, pattern) or fnmatch.fnmatch(rel_str, f"*/{pattern}"):
            matched = True
        # Match against just the filename
        if not anchored and fnmatch.fnmatch(path.name, pattern):
            matched = True
        # Match against any path component (for directory patterns like node_modules/)
        clean = pattern.rstrip("/")
        if not anchored and clean in rel_parts:
            matched = True
        if matched:
            ignored = not negated
    return ignored


def _file_hash(path: Path) -> str:
    """Fast SHA-1 hash of file contents for change detection."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _load_hash_cache(repo_id: str) -> Dict[str, str]:
    """Load previously computed file hashes."""
    cache_file = _hash_cache_file(repo_id)
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())
        except Exception:
            return {}
    return {}


def _save_hash_cache(repo_id: str, cache: Dict[str, str]) -> None:
    """Persist file hashes to disk."""
    _hash_cache_file(repo_id).write_text(json.dumps(cache, indent=2))


def scan_repo(
    repo_root: str | Path,
    incremental: bool = True,
    force_reindex: bool = False,
    include_extensions: set[str] | None = None,
    include_filenames: set[str] | None = None,
    extra_skip_dirs: set[str] | None = None,
    max_file_size_kb: int | None = None,
    update_cache: bool = True,
) -> Tuple[list[ScannedFile], list[str]]:
    """
    Scan a repository and return files to index.

    Args:
        repo_root: Root directory of the repository
        incremental: If True, only return files changed since last scan
        force_reindex: If True, ignore cache and return all files

    Returns:
        (files_to_index, deleted_paths) tuple
    """
    repo_root = Path(repo_root).resolve()
    if not repo_root.exists():
        raise ValueError(f"Repo path does not exist: {repo_root}")
    if not repo_root.is_dir():
        raise ValueError(f"Repo path is not a directory: {repo_root}")

    repo_id = repo_id_for_path(repo_root)
    gitignore_patterns = _load_gitignore_patterns(repo_root)
    hash_cache = {} if force_reindex else _load_hash_cache(repo_id)
    new_cache: Dict[str, str] = {}
    files_to_index: list[ScannedFile] = []
    allowed_exts = include_extensions or set(SUPPORTED_EXTENSIONS)
    allowed_names = include_filenames or set()
    skip_dirs = set(SKIP_DIRS) | (extra_skip_dirs or set())
    max_size_kb = max_file_size_kb or MAX_FILE_SIZE_KB

    print(f"📂 Scanning {repo_root} ...")
    scanned = 0
    skipped = 0

    for dirpath, dirnames, filenames in os.walk(repo_root):
        current_dir = Path(dirpath)

        # Prune directories in-place (modifies os.walk traversal)
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs
            and not _is_gitignored(current_dir / d, repo_root, gitignore_patterns)
        ]

        for filename in filenames:
            if filename in SKIP_FILES:
                skipped += 1
                continue

            filepath = current_dir / filename
            ext = filepath.suffix.lower()

            if filename not in allowed_names and ext not in allowed_exts:
                skipped += 1
                continue

            if ext not in SUPPORTED_EXTENSIONS:
                skipped += 1
                continue

            # Size check
            try:
                size = filepath.stat().st_size
            except OSError:
                continue

            if size > max_size_kb * 1024:
                skipped += 1
                continue

            if _is_gitignored(filepath, repo_root, gitignore_patterns):
                skipped += 1
                continue

            scanned += 1
            rel_path = filepath.relative_to(repo_root).as_posix()

            # Incremental: compute hash and compare
            try:
                current_hash = _file_hash(filepath)
            except OSError:
                continue

            new_cache[rel_path] = current_hash

            if incremental and not force_reindex:
                if hash_cache.get(rel_path) == current_hash:
                    continue  # File unchanged, skip

            files_to_index.append(ScannedFile(
                path=filepath,
                language=SUPPORTED_EXTENSIONS[ext],
                size_bytes=size,
                relative_path=rel_path,
                repo_id=repo_id,
            ))

    # Detect deleted files
    deleted_paths = [p for p in hash_cache if p not in new_cache]

    # Persist updated cache
    if update_cache:
        _save_hash_cache(repo_id, new_cache)

    print(f"   ✅ {scanned} files scanned, {len(files_to_index)} to index, "
          f"{len(deleted_paths)} deleted, {skipped} skipped")

    return files_to_index, deleted_paths
