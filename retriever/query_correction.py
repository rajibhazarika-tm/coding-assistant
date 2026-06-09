# retriever/query_correction.py
"""
Query correction pipeline — runs BEFORE embedding and LLM query understanding.

Handles five categories of query problems that hurt retrieval quality:

  1. Spelling mistakes       "authetication" → "authentication"
  2. CamelCase splitting     "getUserById" → "get user by id"
  3. Abbreviations           "impl" → "implementation", "svc" → "service"
  4. Symbol/operator noise   "user->getName()" → "user getName"
  5. Code mixed with prose   "how does the OrderSvc.processOrder() work" →
                              normalised, with "OrderSvc" → "OrderService"

Design principles:
- Fast: all rules are pure Python, no LLM call for this step (~0ms)
- Lossless: original query is always preserved alongside the corrected one
- Transparent: CorrectedQuery records every change made
- Composable: each corrector is independent, can be toggled
- Fuzzy spelling: uses difflib against an indexed symbol vocabulary so it
  corrects "authentcation" → "authentication" using YOUR codebase's names,
  not a generic dictionary

The corrected query feeds into understand_query() which then does the
deeper LLM-based analysis. Correction happens first so the LLM sees
clean input.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Optional

# ── Common programming abbreviations ─────────────────────────────────────────
# Ordered longest-first so "impl" doesn't interfere with "implementation"
ABBREVIATIONS: dict[str, str] = {
    # Architecture / patterns
    "svc":    "service",
    "repo":   "repository",
    "ctrl":   "controller",
    "mgr":    "manager",
    "impl":   "implementation",
    "iface":  "interface",
    "abs":    "abstract",
    "cfg":    "configuration",
    "config": "configuration",
    "util":   "utility",
    "utils":  "utilities",
    "helper": "helper",
    "hlpr":   "helper",
    "fac":    "factory",
    "bld":    "builder",
    "bldr":   "builder",
    "proc":   "processor",
    "prsr":   "parser",
    "hdlr":   "handler",
    "hndlr":  "handler",
    "mw":     "middleware",
    "mdlwr":  "middleware",
    "orch":   "orchestrator",
    "aggr":   "aggregator",
    # Data / persistence
    "db":     "database",
    "dao":    "data access object",
    "dto":    "data transfer object",
    "orm":    "object relational mapping",
    "qry":    "query",
    "txn":    "transaction",
    "tx":     "transaction",
    "conn":   "connection",
    "ds":     "data source",
    # Auth / security
    "auth":   "authentication",
    "authz":  "authorization",
    "authn":  "authentication",
    "jwt":    "JSON web token",
    "oauth":  "OAuth",
    # HTTP / API
    "req":    "request",
    "resp":   "response",
    "res":    "response",
    "api":    "API",
    "rest":   "REST",
    "rpc":    "remote procedure call",
    "http":   "HTTP",
    "ws":     "web socket",
    # Common ops
    "init":   "initialize",
    "init":   "initialization",
    "exec":   "execute",
    "del":    "delete",
    "upd":    "update",
    "ins":    "insert",
    "msg":    "message",
    "evt":    "event",
    "err":    "error",
    "ex":     "exception",
    "exc":    "exception",
    "val":    "validation",
    "validate": "validation",
    "fmt":    "format",
    "str":    "string",
    "num":    "number",
    "id":     "identifier",
    "pk":     "primary key",
    "fk":     "foreign key",
    # Infra
    "env":    "environment",
    "k8s":    "Kubernetes",
    "mq":     "message queue",
    "q":      "queue",
    "cache":  "cache",
    "cdn":    "content delivery network",
    "lb":     "load balancer",
    "ha":     "high availability",
    "ci":     "continuous integration",
    "cd":     "continuous deployment",
}

# ── Common spelling corrections for programming terms ─────────────────────────
SPELLING_CORRECTIONS: dict[str, str] = {
    "authetication":   "authentication",
    "authentcation":   "authentication",
    "authenticaion":   "authentication",
    "authenication":   "authentication",
    "athentication":   "authentication",
    "authoriation":    "authorization",
    "authorizaion":    "authorization",
    "retreival":       "retrieval",
    "retreive":        "retrieve",
    "recieve":         "receive",
    "reciever":        "receiver",
    "occured":         "occurred",
    "occurence":       "occurrence",
    "seperator":       "separator",
    "seperate":        "separate",
    "dependancy":      "dependency",
    "dependancies":    "dependencies",
    "inheritence":     "inheritance",
    "inheretance":     "inheritance",
    "implemenation":   "implementation",
    "implementaion":   "implementation",
    "implemntation":   "implementation",
    "implemantation":  "implementation",
    "initalize":       "initialize",
    "initalise":       "initialize",
    "initalisation":   "initialization",
    "excecution":      "execution",
    "executoin":       "execution",
    "exceptionn":      "exception",
    "expetion":        "exception",
    "repositry":       "repository",
    "repostiory":      "repository",
    "repositry":       "repository",
    "configration":    "configuration",
    "configurtion":    "configuration",
    "configuartion":   "configuration",
    "valiation":       "validation",
    "validaiton":      "validation",
    "validtion":       "validation",
    "transation":      "transaction",
    "transasction":    "transaction",
    "datatbase":       "database",
    "databse":         "database",
    "dbatase":         "database",
    "connextion":      "connection",
    "conneciton":      "connection",
    "middlewear":      "middleware",
    "middlewhere":     "middleware",
    "contorller":      "controller",
    "cotroller":       "controller",
    "serrvice":        "service",
    "servcie":         "service",
    "managger":        "manager",
    "managment":       "management",
    "mangement":       "management",
    "processer":       "processor",
    "prcessor":        "processor",
    "handeler":        "handler",
    "hnadler":         "handler",
    "interfce":        "interface",
    "inteface":        "interface",
    "funciton":        "function",
    "functon":         "function",
    "fucntion":        "function",
    "methode":         "method",
    "methiod":         "method",
    "paramaeter":      "parameter",
    "paramater":       "parameter",
    "parmaeter":       "parameter",
    "refactoring":     "refactoring",
    "refactring":      "refactoring",
    "inheirtance":     "inheritance",
    "polymorphisim":   "polymorphism",
    "polymorphsim":    "polymorphism",
    "encapuslation":   "encapsulation",
    "encapsualtion":   "encapsulation",
}


@dataclass
class CorrectedQuery:
    """Result of running the correction pipeline on a raw query."""
    original: str          # exactly what the user typed
    corrected: str         # cleaned-up version for LLM + embedding
    corrections: list[str] = field(default_factory=list)  # human-readable log
    symbols_found: list[str] = field(default_factory=list)  # e.g. ["getUserById"]
    was_corrected: bool = False

    @property
    def changed(self) -> bool:
        return self.original.strip() != self.corrected.strip()


# ── Corrector functions ───────────────────────────────────────────────────────

def _split_camel_case(text: str) -> str:
    """
    Split camelCase and PascalCase identifiers into words.
    getUserById  → get user by id
    OrderService → Order Service  (preserves capitals as separate "words")
    HTTPClient   → HTTP Client    (acronyms kept together)
    """
    # Insert space before uppercase that follows lowercase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Insert space before uppercase that is followed by lowercase (after acronym)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', text)
    return text


def _split_snake_case(text: str) -> str:
    """order_service_impl → order service impl"""
    return text.replace('_', ' ')


def _normalise_symbols(text: str) -> str:
    """
    Remove or space-out code punctuation that hurts embedding:
    user.getName() → user getName
    order->process() → order process
    List<String>    → List String
    Map<K,V>        → Map K V
    @Autowired      → Autowired
    #define         → define
    """
    # Strip leading @ and # (annotations, preprocessor)
    text = re.sub(r'[@#](\w)', r'\1', text)
    # Arrow operators → space
    text = re.sub(r'\s*->\s*|\s*=>\s*|\s*::\s*', ' ', text)
    # Dot notation → space (but keep decimal numbers intact)
    text = re.sub(r'(?<!\d)\.(?!\d)', ' ', text)
    # Remove generics/angle brackets
    text = re.sub(r'<[^>]{0,30}>', ' ', text)
    # Remove parentheses and brackets
    text = re.sub(r'[(){}\[\]]', ' ', text)
    # Remove common punctuation except apostrophe
    text = re.sub(r'[;:,!?]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _apply_spelling_corrections(text: str, symbol_vocab: Optional[list[str]] = None) -> tuple[str, list[str]]:
    """
    Apply spelling corrections word by word.

    Two sources:
    1. SPELLING_CORRECTIONS dict (fast, exact)
    2. Fuzzy match against symbol_vocab (from the indexed codebase)
       — only applied to words that look like identifiers (no spaces, >5 chars)
       — cutoff 0.82 to avoid false positives
    """
    words = text.split()
    corrected = []
    changes = []

    for word in words:
        lower = word.lower()

        # Skip very short words, numbers, and already-uppercase acronyms
        if len(word) <= 2 or word.isupper() or word.isdigit():
            corrected.append(word)
            continue

        # 1. Static dictionary
        if lower in SPELLING_CORRECTIONS:
            fixed = SPELLING_CORRECTIONS[lower]
            # Preserve original capitalisation style
            if word[0].isupper():
                fixed = fixed[0].upper() + fixed[1:]
            if fixed.lower() != lower:
                changes.append(f"'{word}' → '{fixed}'")
            corrected.append(fixed)
            continue

        # 2. Fuzzy match against codebase symbols (only for longer words)
        # Skip if vocab is empty or word is too short — avoids O(n) scan
        if symbol_vocab and len(word) >= 7:
            matches = get_close_matches(lower, [s.lower() for s in symbol_vocab],
                                        n=1, cutoff=0.85)
            if matches:
                # Find the original casing from symbol_vocab
                vocab_lower = [s.lower() for s in symbol_vocab]
                idx = vocab_lower.index(matches[0])
                fixed = symbol_vocab[idx]
                if fixed.lower() != lower:
                    changes.append(f"'{word}' → '{fixed}' (codebase symbol)")
                    corrected.append(fixed)
                    continue

        corrected.append(word)

    return " ".join(corrected), changes


def _expand_abbreviations(text: str) -> tuple[str, list[str]]:
    """
    Expand common programming abbreviations.
    Whole-word match only — 'svc' matches but 'service' does not get re-expanded.
    """
    words = text.split()
    result = []
    changes = []

    for word in words:
        lower = word.lower()
        # Strip trailing punctuation for lookup
        stripped = lower.rstrip(string.punctuation)

        if stripped in ABBREVIATIONS:
            expanded = ABBREVIATIONS[stripped]
            # Preserve capitalisation of first letter
            if word[0].isupper():
                expanded = expanded[0].upper() + expanded[1:]
            if expanded.lower() != lower:
                changes.append(f"'{word}' → '{expanded}'")
            result.append(expanded)
        else:
            result.append(word)

    return " ".join(result), changes


def _extract_code_symbols(text: str) -> list[str]:
    """
    Pull out identifier-like tokens (camelCase, PascalCase, snake_case)
    from the query for use as grep search terms.
    These are usually the most precise part of a query.
    """
    symbols = []
    # camelCase / PascalCase identifiers
    symbols += re.findall(r'\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b', text)
    symbols += re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', text)
    # snake_case identifiers (at least one underscore)
    symbols += re.findall(r'\b\w+_\w+\b', text)
    # Method-like patterns: word()
    symbols += [m.rstrip('()') for m in re.findall(r'\b\w+\(\)', text)]
    # Deduplicate preserving order
    seen: set[str] = set()
    unique = []
    for s in symbols:
        if s not in seen and len(s) > 3:
            seen.add(s); unique.append(s)
    return unique


def _load_symbol_vocab(max_symbols: int = 500) -> list[str]:
    """
    Load symbol names from the ChromaDB index for fuzzy spelling correction.

    Capped at 500 symbols — more than enough for close-match correction,
    and avoids O(n) difflib scan over 77k symbols per query word.
    Only loads names ≥ 5 chars (short names cause too many false positives
    and aren't worth the fuzzy match cost).
    """
    try:
        from indexer.embedder import _get_chroma_client
        from config.settings import CHROMA_COLLECTION
        client = _get_chroma_client()
        col = client.get_collection(CHROMA_COLLECTION)
        if col.count() == 0:
            return []
        result = col.get(limit=min(max_symbols * 3, col.count()),
                         include=["metadatas"])
        names = set()
        for meta in result.get("metadatas", []):
            if meta:
                name = meta.get("name", "")
                # Only keep meaningful identifiers (≥5 chars, not purely numeric)
                if name and len(name) >= 5 and not name.isdigit():
                    names.add(name)
                    if len(names) >= max_symbols:
                        break
        return list(names)
    except Exception:
        return []


# ── Vocabulary cache ──────────────────────────────────────────────────────────
_vocab_cache: list[str] = []
_vocab_loaded: bool = False


def _get_vocab(refresh: bool = False) -> list[str]:
    """Return cached symbol vocabulary, loading it once on first call."""
    global _vocab_cache, _vocab_loaded
    if not _vocab_loaded or refresh:
        _vocab_cache = _load_symbol_vocab()
        _vocab_loaded = True
    return _vocab_cache


# ── Main pipeline ─────────────────────────────────────────────────────────────

def correct_query(
    question: str,
    use_vocab: bool = True,
    expand_abbrevs: bool = True,
    fix_spelling: bool = True,
    split_identifiers: bool = True,
) -> CorrectedQuery:
    """
    Run the full query correction pipeline.

    Steps (all pure Python, <1ms total):
      1. Extract code symbols from the raw query (preserved for grep)
      2. Normalise punctuation / operators
      3. Split camelCase / snake_case identifiers
      4. Fix spelling (static dict + optional fuzzy codebase vocab)
      5. Expand abbreviations
      6. Final cleanup

    Returns a CorrectedQuery with the cleaned text and a log of changes.
    The original is always preserved — correction never loses information.
    """
    if not question or not question.strip():
        return CorrectedQuery(original=question, corrected=question)

    text = question.strip()
    all_changes: list[str] = []

    # Step 1: extract raw symbols before we modify the text
    symbols = _extract_code_symbols(text)

    # Step 2: normalise punctuation (arrows, dots, parens, angle brackets)
    text = _normalise_symbols(text)

    # Step 3: split identifiers
    if split_identifiers:
        prev = text
        # First split camelCase/PascalCase
        parts = []
        for word in text.split():
            if re.search(r'[a-z][A-Z]|[A-Z]{2}[a-z]', word):
                parts.append(_split_camel_case(word))
            else:
                parts.append(word)
        text = " ".join(parts)
        # Then split snake_case
        if '_' in text:
            text = _split_snake_case(text)
        if text != prev:
            all_changes.append(f"identifier split")

    # Step 4: spelling correction
    if fix_spelling:
        vocab = _get_vocab() if use_vocab else []
        text, spell_changes = _apply_spelling_corrections(text, vocab)
        all_changes.extend(spell_changes)

    # Step 5: abbreviation expansion
    if expand_abbrevs:
        text, abbrev_changes = _expand_abbreviations(text)
        all_changes.extend(abbrev_changes)

    # Step 6: final cleanup — collapse spaces, fix capitalisation
    text = re.sub(r'\s+', ' ', text).strip()

    # Preserve sentence-starting capital if original had it
    if question[0].isupper() and text and text[0].islower():
        text = text[0].upper() + text[1:]

    return CorrectedQuery(
        original=question,
        corrected=text,
        corrections=all_changes,
        symbols_found=symbols,
        was_corrected=bool(all_changes),
    )


def invalidate_vocab_cache() -> None:
    """Call after indexing completes so the symbol vocab is refreshed."""
    global _vocab_loaded
    _vocab_loaded = False
