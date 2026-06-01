# indexer/strategy.py
"""
Automatic indexing strategy selection for large and mixed repositories.
"""

from __future__ import annotations

import os
import sys

# FIX 8: Path.is_relative_to() requires Python 3.9+
if sys.version_info < (3, 9):
    raise RuntimeError(
        "coding-assistant requires Python 3.9 or later "
        f"(you have {sys.version_info.major}.{sys.version_info.minor}). "
        "Please upgrade your Python installation."
    )
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import SKIP_DIRS, SUPPORTED_EXTENSIONS


@dataclass(frozen=True)
class IndexProfile:
    name: str
    description: str
    include_extensions: set[str]
    include_filenames: set[str] = field(default_factory=set)
    extra_skip_dirs: set[str] = field(default_factory=set)
    max_file_size_kb: int = 500


@dataclass
class PlanArea:
    path: Path
    profile: str
    reason: str
    estimated_files: int = 0
    estimated_bytes: int = 0


@dataclass
class RepoAnalysis:
    root: Path
    repo_type: str
    total_files_seen: int
    total_bytes_seen: int
    markers: list[str]
    areas: list[PlanArea]
    skipped_dirs: list[str]


COMMON_SKIP_DIRS = {
    ".git", ".svn", ".hg", ".idea", ".vscode",
    "node_modules", "target", "build", "dist", "out",
    ".gradle", ".mvn", "__pycache__", ".pytest_cache",
    "coverage", "logs", "log", "tmp", "temp",
    "generated", "generated-sources", "generated-test-sources",
}


PROFILES: dict[str, IndexProfile] = {
    "generic": IndexProfile(
        name="generic",
        description="General source-code profile.",
        include_extensions=set(SUPPORTED_EXTENSIONS),
        extra_skip_dirs=COMMON_SKIP_DIRS,
    ),
    "spring": IndexProfile(
        name="spring",
        description="Java/Spring Boot backend profile.",
        include_extensions={".java", ".kt", ".xml", ".yml", ".yaml", ".properties", ".gradle", ".md", ".json", ".sql"},
        include_filenames={"pom.xml", "build.gradle", "settings.gradle", "gradle.properties"},
        extra_skip_dirs=COMMON_SKIP_DIRS,
    ),
    "java-library": IndexProfile(
        name="java-library",
        description="Java/Kotlin framework or shared library profile.",
        include_extensions={".java", ".kt", ".xml", ".yml", ".yaml", ".properties", ".gradle", ".md", ".json"},
        include_filenames={"pom.xml", "build.gradle", "settings.gradle", "gradle.properties"},
        extra_skip_dirs=COMMON_SKIP_DIRS,
    ),
    "frontend": IndexProfile(
        name="frontend",
        description="Frontend UI profile.",
        include_extensions={".js", ".jsx", ".ts", ".tsx", ".json", ".md", ".yaml", ".yml", ".html", ".css", ".scss"},
        include_filenames={"package.json", "tsconfig.json", "angular.json", "vite.config.js", "vite.config.ts", "next.config.js", "next.config.ts"},
        extra_skip_dirs=COMMON_SKIP_DIRS,
        max_file_size_kb=350,
    ),
}


def get_profile(name: str | None) -> IndexProfile:
    if not name or name == "auto":
        return PROFILES["generic"]
    if name not in PROFILES:
        raise ValueError(f"Unknown profile: {name}")
    return PROFILES[name]


def analyze_repo(repo_root: str | Path, max_files: int = 250_000) -> RepoAnalysis:
    """Inspect a repo cheaply and choose indexing areas/profiles.

    FIX 7: was walking the tree twice (once here, once per area in _estimate_areas).
    Now accumulates per-directory file/byte counts during this single pass so
    _estimate_areas_fast can answer by summing the pre-built map instead of re-walking.
    """
    root = Path(repo_root).resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"Repo path is not a directory: {root}")

    markers: set[str] = set()
    skipped_dirs: set[str] = set()
    candidates: dict[Path, set[str]] = {}
    total_files = 0
    total_bytes = 0
    # dir_stats[dir_path][ext_or_filename] = (file_count, byte_count)
    dir_stats: dict[Path, dict[str, tuple[int, int]]] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        original_dirs = list(dirnames)
        dirnames[:] = [d for d in dirnames if d not in COMMON_SKIP_DIRS and d not in SKIP_DIRS]
        skipped_dirs.update(d for d in original_dirs if d not in dirnames)

        rel_dir = current.relative_to(root).as_posix() if current != root else "."
        filename_set = set(filenames)
        dir_set = set(dirnames)

        local_markers = _detect_markers(rel_dir, filename_set, dir_set)
        if local_markers:
            markers.update(local_markers)
            candidates.setdefault(current, set()).update(local_markers)

        dir_entry: dict[str, tuple[int, int]] = {}
        for filename in filenames:
            total_files += 1
            try:
                size = (current / filename).stat().st_size
                total_bytes += size
            except OSError:
                size = 0
            key = Path(filename).suffix.lower() or filename
            fc, bc = dir_entry.get(key, (0, 0))
            dir_entry[key] = (fc + 1, bc + size)
            if total_files >= max_files:
                markers.add("scan-limit-reached")
                break
        dir_stats[current] = dir_entry
        if total_files >= max_files:
            break

    areas = _choose_areas(root, candidates)
    if not areas:
        areas = [PlanArea(path=root, profile="generic", reason="No framework markers found; using generic source profile.")]

    _estimate_areas_fast(areas, dir_stats)
    repo_type = _repo_type_for_areas(areas)
    return RepoAnalysis(
        root=root,
        repo_type=repo_type,
        total_files_seen=total_files,
        total_bytes_seen=total_bytes,
        markers=sorted(markers),
        areas=areas,
        skipped_dirs=sorted(skipped_dirs),
    )


def format_analysis(analysis: RepoAnalysis) -> str:
    """Human-readable analysis summary for CLI output."""
    lines = [
        f"Repo: {analysis.root}",
        f"Detected type: {analysis.repo_type}",
        f"Files sampled: {analysis.total_files_seen:,}",
        f"Approx bytes sampled: {analysis.total_bytes_seen:,}",
    ]
    if analysis.markers:
        lines.append("Markers: " + ", ".join(analysis.markers[:12]))
    if analysis.skipped_dirs:
        lines.append("Skip dirs seen: " + ", ".join(analysis.skipped_dirs[:12]))
    lines.append("")
    lines.append("Index plan:")
    for area in analysis.areas:
        rel = area.path.relative_to(analysis.root).as_posix() if area.path != analysis.root else "."
        lines.append(
            f"  - {rel} -> {area.profile} "
            f"({area.estimated_files:,} files, {area.estimated_bytes:,} bytes): {area.reason}"
        )
    return "\n".join(lines)


def _detect_markers(rel_dir: str, filenames: set[str], dirnames: set[str]) -> set[str]:
    markers: set[str] = set()
    lower_files = {f.lower() for f in filenames}

    if "pom.xml" in lower_files:
        markers.add("maven")
    if "build.gradle" in lower_files or "settings.gradle" in lower_files:
        markers.add("gradle")
    if "src" in dirnames and ("pom.xml" in lower_files or "build.gradle" in lower_files):
        markers.add("java-project")
    if rel_dir.endswith("src/main") and "java" in dirnames:
        markers.add("spring-layout")
    if "application.yml" in lower_files or "application.yaml" in lower_files or "application.properties" in lower_files:
        markers.add("spring-config")
    if "package.json" in lower_files:
        markers.add("node-ui")
    if "angular.json" in lower_files:
        markers.add("angular")
    if any(f.startswith("vite.config") for f in lower_files):
        markers.add("vite")
    if any(f.startswith("next.config") for f in lower_files):
        markers.add("next")
    return markers


def _choose_areas(root: Path, candidates: dict[Path, set[str]]) -> list[PlanArea]:
    areas: list[PlanArea] = []
    for path, markers in sorted(candidates.items(), key=lambda item: len(item[0].parts)):
        if _covered_by_existing(path, areas):
            continue
        profile = _profile_for_markers(path, markers)
        reason = "Detected " + ", ".join(sorted(markers))
        areas.append(PlanArea(path=path, profile=profile, reason=reason))

    # Prefer obvious top-level domains when present, even if their marker is nested.
    named_roots = []
    for name, profile in (("backend", "spring"), ("framework", "java-library"), ("ui", "frontend"), ("frontend", "frontend")):
        path = root / name
        if path.exists() and path.is_dir() and not _covered_by_existing(path, areas):
            named_roots.append(PlanArea(path=path, profile=profile, reason=f"Detected conventional {name}/ area."))
    return named_roots + areas


def _covered_by_existing(path: Path, areas: list[PlanArea]) -> bool:
    return any(path == area.path or path.is_relative_to(area.path) for area in areas)


def _profile_for_markers(path: Path, markers: set[str]) -> str:
    path_text = path.as_posix().lower()
    if markers & {"angular", "vite", "next", "node-ui"} or path.name.lower() in {"ui", "frontend", "web"}:
        return "frontend"
    if "framework" in path_text:
        return "java-library"
    if markers & {"spring-config", "spring-layout"} or path.name.lower() in {"backend", "server", "service"}:
        return "spring"
    if markers & {"maven", "gradle", "java-project"}:
        return "java-library"
    return "generic"


def _estimate_areas(areas: list[PlanArea]) -> None:
    """Legacy: full os.walk per area. Kept for callers outside analyze_repo."""
    for area in areas:
        profile = get_profile(area.profile)
        files = 0
        bytes_seen = 0
        for dirpath, dirnames, filenames in os.walk(area.path):
            dirnames[:] = [d for d in dirnames if d not in profile.extra_skip_dirs and d not in SKIP_DIRS]
            for filename in filenames:
                ext = Path(filename).suffix.lower()
                if filename not in profile.include_filenames and ext not in profile.include_extensions:
                    continue
                if ext not in SUPPORTED_EXTENSIONS:
                    continue
                file_path = Path(dirpath) / filename
                try:
                    size = file_path.stat().st_size
                except OSError:
                    continue
                if size > profile.max_file_size_kb * 1024:
                    continue
                files += 1
                bytes_seen += size
        area.estimated_files = files
        area.estimated_bytes = bytes_seen


def _estimate_areas_fast(
    areas: list[PlanArea],
    dir_stats: dict,
) -> None:
    """FIX 7: estimate file/byte counts from the pre-built dir_stats map
    accumulated during the first os.walk in analyze_repo — no second walk needed."""
    for area in areas:
        profile = get_profile(area.profile)
        include_exts = profile.include_extensions
        include_names = profile.include_filenames
        max_bytes = profile.max_file_size_kb * 1024
        skip = profile.extra_skip_dirs | SKIP_DIRS
        files = 0
        bytes_seen = 0
        for dir_path, entry in dir_stats.items():
            # Only count directories that are under this area and not in skip dirs
            try:
                rel = dir_path.relative_to(area.path)
            except ValueError:
                continue
            if any(part in skip for part in rel.parts):
                continue
            for key, (fc, bc) in entry.items():
                # key is an extension (.java) or bare filename (pom.xml)
                if key not in include_exts and key not in include_names:
                    continue
                if key not in SUPPORTED_EXTENSIONS and key not in include_names:
                    continue
                # byte filter: approximate — skip only if avg file > max_bytes
                avg = bc // max(fc, 1)
                if avg > max_bytes:
                    continue
                files += fc
                bytes_seen += bc
        area.estimated_files = files
        area.estimated_bytes = bytes_seen


def _repo_type_for_areas(areas: list[PlanArea]) -> str:
    profiles = {area.profile for area in areas}
    if len(profiles) > 1:
        return "mixed-monorepo"
    if "spring" in profiles:
        return "spring-backend"
    if "frontend" in profiles:
        return "frontend-ui"
    if "java-library" in profiles:
        return "java-framework"
    return "generic-codebase"
