# indexer/chunker.py
"""
AST-aware code chunker using tree-sitter.

WHY AST CHUNKING?
- Naive line/character splitting cuts functions in half → broken context
- AST chunks preserve logical units: functions, classes, methods
- Each chunk is semantically complete and embeddable on its own
- Proven to dramatically improve retrieval quality vs. text splitting

STRATEGY:
1. Parse file with tree-sitter → concrete syntax tree
2. Extract top-level declarations (functions, classes, etc.)
3. If a node is too large, split at its children (e.g., class methods)
4. Fall back to line-based windowing for unsupported languages
"""

from __future__ import annotations
import ast
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from config.settings import CHUNK_MAX_LINES, CHUNK_MIN_LINES


@dataclass
class CodeChunk:
    """A single semantic unit of code ready for embedding."""
    content: str               # The actual code text
    file_path: str             # Relative path in repo
    language: str              # e.g. "python", "javascript"
    start_line: int            # 1-indexed
    end_line: int
    chunk_type: str            # "function", "class", "method", "block", "file"
    name: Optional[str]        # Symbol name if available (e.g. "parse_config")
    context_header: str = ""   # e.g. "class MyClass > def my_method"
    repo_id: str = ""          # Stable repository identifier for multi-repo indexes

    @property
    def id(self) -> str:
        """Stable unique ID for this chunk."""
        base = f"{self.file_path}:{self.start_line}-{self.end_line}"
        return f"{self.repo_id}:{base}" if self.repo_id else base

    @property
    def embedding_text(self) -> str:
        """Text to embed: context header + code for better retrieval."""
        parts = []
        if self.context_header:
            parts.append(f"# {self.context_header}")
        parts.append(f"# File: {self.file_path}")
        parts.append(self.content)
        return "\n".join(parts)


# ─── tree-sitter node types that represent top-level declarations ─────────────

_TOP_LEVEL_TYPES: dict[str, list[str]] = {
    "python": [
        "function_definition", "async_function_definition",
        "class_definition", "decorated_definition",
    ],
    "javascript": [
        "function_declaration", "arrow_function",
        "class_declaration", "method_definition",
        "export_statement", "lexical_declaration",
    ],
    "typescript": [
        "function_declaration", "arrow_function",
        "class_declaration", "method_definition",
        "interface_declaration", "type_alias_declaration",
        "export_statement",
    ],
    "tsx": [
        "function_declaration", "arrow_function",
        "class_declaration", "jsx_element",
        "export_statement",
    ],
    "java": [
        "class_declaration", "interface_declaration",
        "method_declaration", "constructor_declaration",
    ],
    "go": [
        "function_declaration", "method_declaration",
        "type_declaration", "var_declaration",
    ],
    "rust": [
        "function_item", "impl_item", "struct_item",
        "trait_item", "enum_item", "mod_item",
    ],
    "cpp": [
        "function_definition", "class_specifier",
        "namespace_definition", "declaration",
    ],
    "c": ["function_definition", "declaration", "struct_specifier"],
    "c_sharp": [
        "class_declaration", "interface_declaration",
        "method_declaration", "constructor_declaration",
        "namespace_declaration",
    ],
    "ruby": [
        "method", "singleton_method", "class", "module",
    ],
}

_NAME_FIELDS = ["name", "identifier"]


def _try_import_treesitter(language: str):
    """Attempt to import tree-sitter parser for a language. Returns (parser, None) or (None, error)."""
    try:
        import tree_sitter_languages  # type: ignore
        parser = tree_sitter_languages.get_parser(language)
        return parser, None
    except Exception as e:
        return None, str(e)


def _node_name(node, _depth: int = 0) -> Optional[str]:
    """Extract the declaration name from a tree-sitter node.

    Only recurses one level deep into wrapper nodes (decorated_definition,
    export_statement) to avoid returning an identifier from the function body.
    FIX 5: was unbounded; now capped at depth=1.
    """
    # 1. Try named field first (most reliable)
    for field_name in _NAME_FIELDS:
        child = node.child_by_field_name(field_name)
        if child:
            return child.text.decode("utf-8", errors="replace")

    # 2. Try direct children whose type IS a name
    for child in node.children:
        if child.type in _NAME_FIELDS or child.type == "name":
            return child.text.decode("utf-8", errors="replace")

    # 3. One level of recursion only — for wrapper nodes like decorated_definition
    #    or export_statement that contain the real declaration as a child.
    if _depth == 0:
        for child in node.children:
            nested = _node_name(child, _depth=1)
            if nested:
                return nested

    return None


def _lines_of(node, source_lines: list[str]) -> list[str]:
    start = node.start_point[0]
    end = node.end_point[0] + 1
    return source_lines[start:end]


def _chunk_node(
    node,
    source_lines: list[str],
    file_path: str,
    language: str,
    repo_id: str = "",
    parent_name: str = "",
) -> list[CodeChunk]:
    """Recursively chunk a node, splitting at children if too large."""
    start_line = node.start_point[0]
    end_line = node.end_point[0]
    num_lines = end_line - start_line + 1
    lines = source_lines[start_line: end_line + 1]

    if num_lines < CHUNK_MIN_LINES:
        return []

    name = _node_name(node)
    context = f"{parent_name} > {name}" if parent_name and name else (name or node.type)

    if num_lines <= CHUNK_MAX_LINES:
        return [CodeChunk(
            content="\n".join(lines),
            file_path=file_path,
            language=language,
            start_line=start_line + 1,
            end_line=end_line + 1,
            chunk_type=node.type,
            name=name,
            context_header=context,
            repo_id=repo_id,
        )]

    # Node too large — split into children
    chunks = []
    top_types = _TOP_LEVEL_TYPES.get(language, [])
    child_chunks_found = False

    for child in node.children:
        if child.type in top_types or child.child_count > 3:
            sub = _chunk_node(child, source_lines, file_path, language, repo_id=repo_id, parent_name=context)
            if sub:
                chunks.extend(sub)
                child_chunks_found = True

    if not child_chunks_found:
        # Fallback: sliding window over lines
        chunks.extend(_line_window_chunks(lines, start_line, file_path, language, context, repo_id=repo_id))

    return chunks


def _line_window_chunks(
    lines: list[str],
    start_offset: int,
    file_path: str,
    language: str,
    context: str,
    repo_id: str = "",
    window: int = CHUNK_MAX_LINES,
    overlap: int = 10,
) -> list[CodeChunk]:
    """Fallback: sliding window when AST node is too large and has no sub-nodes."""
    chunks = []
    i = 0
    while i < len(lines):
        window_lines = lines[i: i + window]
        if len(window_lines) < CHUNK_MIN_LINES:
            break
        start = start_offset + i + 1
        end = start + len(window_lines) - 1
        chunks.append(CodeChunk(
            content="\n".join(window_lines),
            file_path=file_path,
            language=language,
            start_line=start,
            end_line=end,
            chunk_type="block",
            name=None,
            context_header=context,
            repo_id=repo_id,
        ))
        i += window - overlap  # overlap for continuity
    return chunks


def _python_ast_chunks(file_path: str, content: str, repo_id: str = "") -> list[CodeChunk]:
    """Chunk Python with the stdlib AST when tree-sitter is unavailable."""
    source_lines = content.splitlines()
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    chunks: list[CodeChunk] = []

    def add_node(node: ast.AST, parent_name: str = "") -> None:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return
        start_line = int(getattr(node, "lineno"))
        end_line = int(getattr(node, "end_lineno") or start_line)
        num_lines = end_line - start_line + 1
        if num_lines < CHUNK_MIN_LINES:
            return

        name = getattr(node, "name", None)
        context = f"{parent_name} > {name}" if parent_name and name else (name or type(node).__name__)

        if num_lines <= CHUNK_MAX_LINES:
            content_slice = "\n".join(source_lines[start_line - 1:end_line])
            chunks.append(CodeChunk(
                content=content_slice,
                file_path=file_path,
                language="python",
                start_line=start_line,
                end_line=end_line,
                chunk_type=type(node).__name__,
                name=name,
                context_header=context,
                repo_id=repo_id,
            ))
            return

        child_nodes = [
            child for child in getattr(node, "body", [])
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        if child_nodes:
            for child in child_nodes:
                add_node(child, parent_name=context)
        else:
            chunks.extend(_line_window_chunks(
                source_lines[start_line - 1:end_line],
                start_line - 1,
                file_path,
                "python",
                context,
                repo_id=repo_id,
            ))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add_node(node)

    return chunks


def chunk_file(file_path: str, language: str, content: str, repo_id: str = "") -> list[CodeChunk]:
    """
    Main entry point: parse a file and return semantic chunks.

    Tries tree-sitter first; falls back to line windowing if unavailable.
    """
    source_lines = content.splitlines()

    if not source_lines:
        return []

    if language == "python":
        chunks = _python_ast_chunks(file_path, content, repo_id=repo_id)
        if chunks:
            return chunks

    # Try AST parsing
    parser, err = _try_import_treesitter(language)
    if parser is not None:
        try:
            tree = parser.parse(content.encode("utf-8", errors="replace"))
            root = tree.root_node
            top_types = _TOP_LEVEL_TYPES.get(language, [])
            chunks = []

            for child in root.children:
                if child.type in top_types:
                    chunks.extend(_chunk_node(child, source_lines, file_path, language, repo_id=repo_id))

            # If nothing found (e.g. script-style file with no top-level defs),
            # fall back to windowing the whole file
            if not chunks:
                chunks = _line_window_chunks(source_lines, 0, file_path, language, file_path, repo_id=repo_id)

            return chunks

        except Exception:
            pass  # Fall through to line-based fallback

    # Fallback: line-based windowing
    return _line_window_chunks(source_lines, 0, file_path, language, file_path, repo_id=repo_id)
