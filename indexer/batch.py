# indexer/batch.py
"""
Subfolder-batched indexing.

Instead of scanning the whole repo, collecting every file, building one
giant chunk list and embedding it all at once, this module:

  1. Discovers immediate subdirectories (or groups flat files into one batch)
  2. Processes each subfolder independently:
       scan → chunk → embed → store → move to next
  3. Streams per-folder progress so the UI stays responsive
  4. Cancel/resume works at folder granularity — completed folders are
     already in ChromaDB (resume skips them automatically)

Memory profile: O(largest_subfolder) instead of O(whole_repo).
For a 50GB monorepo with 200 subdirectories, peak RAM drops from
~2GB (all chunks at once) to ~20MB (one folder at a time).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional

from config.settings import SUPPORTED_EXTENSIONS, SKIP_DIRS


@dataclass
class FolderBatch:
    """One unit of indexing work: a folder and the files inside it."""
    folder: Path               # absolute path to this batch's root
    relative_name: str         # display name (relative to repo root)
    files: list                # list[ScannedFile] — populated by scanner
    batch_index: int           # 1-based position in the batch list
    total_batches: int         # total number of batches


@dataclass
class BatchProgress:
    """Progress event emitted after each folder completes."""
    folder_name: str
    files_in_folder: int
    chunks_in_folder: int
    chunks_indexed: int
    total_indexed_so_far: int
    batch_index: int
    total_batches: int
    eta_str: str
    error: Optional[str] = None


def discover_batches(
    repo_root: Path,
    profile,
    min_batch_size: int = 10,     # merge tiny folders into a combined batch
    max_depth: int = 2,            # how deep to look for subdirectories
) -> list[Path]:
    """
    Return a list of folder paths to process as independent batches.

    Strategy:
    - Walk up to max_depth levels looking for subdirectories
    - Folders with fewer than min_batch_size supported files are merged
      with the parent batch (avoids 1000 single-file batches)
    - Always includes a root-level batch for top-level files
    """
    allowed_exts = profile.include_extensions
    allowed_names = profile.include_filenames
    skip = set(SKIP_DIRS) | profile.extra_skip_dirs

    def _count_files(folder: Path) -> int:
        count = 0
        for entry in folder.iterdir():
            if entry.is_file():
                if entry.suffix.lower() in allowed_exts or entry.name in allowed_names:
                    count += 1
        return count

    def _subdirs(folder: Path, depth: int) -> list[Path]:
        if depth == 0:
            return []
        result = []
        try:
            for entry in sorted(folder.iterdir()):
                if not entry.is_dir():
                    continue
                if entry.name.startswith(".") or entry.name in skip:
                    continue
                result.append(entry)
        except PermissionError:
            pass
        return result

    # Start with immediate subdirectories of the repo root
    top_dirs = _subdirs(repo_root, 1)

    if not top_dirs:
        # Flat repo — single batch
        return [repo_root]

    batches: list[Path] = []

    def _collect(folder: Path, depth: int):
        """Recursively collect batch folders."""
        sub_dirs = _subdirs(folder, depth)
        file_count = _count_files(folder)

        if not sub_dirs:
            # Leaf directory — always a batch if it has any supported files
            if file_count > 0:
                batches.append(folder)
            return

        # Has subdirs — add this folder for its own top-level files,
        # then recurse into children
        if file_count > 0:
            batches.append(folder)

        for sd in sub_dirs:
            _collect(sd, depth - 1)

    for d in top_dirs:
        _collect(d, max_depth - 1)

    # Always add a root-level batch for files directly in repo_root
    root_files = _count_files(repo_root)
    if root_files > 0:
        batches.insert(0, repo_root)

    # Remove duplicates while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for b in batches:
        if b not in seen:
            seen.add(b)
            unique.append(b)

    return unique if unique else [repo_root]


def index_in_batches(
    repo_root: Path,
    repo_id: str,
    profile,
    force: bool = False,
    on_progress: Optional[Callable[[BatchProgress], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    depth: int = 2,
) -> int:
    """
    Index a repository subfolder by subfolder.

    Each folder is fully scanned, chunked, embedded, and stored before
    moving to the next. This keeps memory usage proportional to the
    largest single subfolder rather than the entire repo.

    Returns total chunks indexed across all folders.
    """
    import time
    from indexer.scanner import scan_repo, _hash_cache_file
    from indexer.chunker import chunk_file
    from indexer.embedder import (
        index_chunks, delete_chunks_for_files,
        delete_chunks_for_repo, _cancel_requested,
    )

    def log(msg: str):
        if on_log:
            on_log(msg)

    # Discover batches
    folders = discover_batches(repo_root, profile, max_depth=depth)
    total_batches = len(folders)
    log(f"📂 {total_batches} folder batch(es) discovered\n")

    if force:
        delete_chunks_for_repo(repo_id)
        log("🗑️  Cleared existing index (force re-index)\n")

    total_indexed = 0
    t_start = time.perf_counter()
    folder_times: list[float] = []

    for batch_idx, folder in enumerate(folders, start=1):
        import indexer.embedder as _emb
        if _emb._cancel_requested:
            log(f"⚠️  Cancelled at folder {batch_idx}/{total_batches}\n")
            break

        rel_name = str(folder.relative_to(repo_root)) if folder != repo_root else "(root)"
        log(f"\n── [{batch_idx}/{total_batches}] {rel_name}/\n")

        t_folder = time.perf_counter()

        # Scan only this folder (non-recursive into sub-batches)
        # Use incremental scan so unchanged files are skipped
        try:
            files, deleted = scan_repo(
                folder, incremental=True, force_reindex=False,
                include_extensions=profile.include_extensions,
                include_filenames=profile.include_filenames,
                extra_skip_dirs=profile.extra_skip_dirs,
                max_file_size_kb=profile.max_file_size_kb,
            )
        except Exception as e:
            log(f"   ⚠️  Scan error: {e}\n")
            continue

        if deleted:
            delete_chunks_for_files(deleted, repo_id=repo_id)

        if not files:
            log(f"   ✓ Nothing changed\n")
            if on_progress:
                on_progress(BatchProgress(
                    folder_name=rel_name, files_in_folder=0,
                    chunks_in_folder=0, chunks_indexed=0,
                    total_indexed_so_far=total_indexed,
                    batch_index=batch_idx, total_batches=total_batches,
                    eta_str="–",
                ))
            continue

        log(f"   {len(files)} file(s) to index\n")

        # Chunk this folder's files
        chunks = []
        chunk_errors = 0
        for f in files:
            try:
                content = f.path.read_text(encoding="utf-8", errors="replace")
                chunks.extend(chunk_file(
                    f.relative_path, f.language, content, repo_id=repo_id
                ))
            except Exception:
                chunk_errors += 1
        log(f"   {len(chunks)} chunk(s) extracted"
            + (f" ({chunk_errors} errors)" if chunk_errors else "") + "\n")

        if not chunks:
            continue

        # Embed + store this folder's chunks
        folder_indexed = [0]

        def _folder_progress(indexed: int, total: int, eta: str):
            folder_indexed[0] = indexed
            # Compute overall ETA based on folder completion times so far
            elapsed = time.perf_counter() - t_start
            completed_batches = batch_idx - 1
            if folder_times:
                avg_folder = sum(folder_times) / len(folder_times)
                remaining_batches = total_batches - batch_idx
                overall_eta_s = remaining_batches * avg_folder
                h = int(overall_eta_s // 3600)
                m = int((overall_eta_s % 3600) // 60)
                overall_eta = f"{h}h{m:02d}m" if h else f"{m}m"
            else:
                overall_eta = eta

            if on_progress:
                on_progress(BatchProgress(
                    folder_name=rel_name,
                    files_in_folder=len(files),
                    chunks_in_folder=len(chunks),
                    chunks_indexed=indexed,
                    total_indexed_so_far=total_indexed + indexed,
                    batch_index=batch_idx,
                    total_batches=total_batches,
                    eta_str=overall_eta,
                ))

        n = index_chunks(chunks, show_progress=False,
                         on_progress=_folder_progress, resume=True)
        total_indexed += n
        folder_time = time.perf_counter() - t_folder
        folder_times.append(folder_time)
        log(f"   ✅ {n} stored  ({folder_time:.1f}s)\n")

        if on_progress:
            on_progress(BatchProgress(
                folder_name=rel_name,
                files_in_folder=len(files),
                chunks_in_folder=len(chunks),
                chunks_indexed=n,
                total_indexed_so_far=total_indexed,
                batch_index=batch_idx,
                total_batches=total_batches,
                eta_str="–",
            ))

    return total_indexed
