#!/usr/bin/env python3
"""
FastAPI backend for the coding assistant UI.
All CLI operations exposed as REST endpoints + SSE streaming for indexing.
"""
from __future__ import annotations
import asyncio, json, os, sys, time
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))

app = FastAPI(title="Coding Assistant API", version="2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Request models ────────────────────────────────────────────────────────────
class IndexRequest(BaseModel):
    path: str
    force: bool = False
    auto: bool = True
    profile: str = "generic"

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    language_filter: Optional[str] = None
    repo_root: Optional[str] = None
    use_pipeline: bool = True           # False = legacy hybrid-only
    use_rag: bool = True                # True = full RAG pipeline
    use_compression: bool = False       # True = contextual compression (slower)
    check_faithfulness: bool = False    # True = verify answer vs context

class ReviewRequest(BaseModel):
    file_path: str

class ExplainRequest(BaseModel):
    file_path: Optional[str] = None
    function_name: Optional[str] = None

class GenerateRequest(BaseModel):
    description: str

class SettingsUpdate(BaseModel):
    model: Optional[str] = None
    embed_model: Optional[str] = None
    ollama_base_url: Optional[str] = None
    max_context_tokens: Optional[int] = None
    top_k_chunks: Optional[int] = None
    chunk_max_lines: Optional[int] = None
    embed_workers: Optional[int] = None
    chroma_batch_size: Optional[int] = None
    embed_num_ctx: Optional[int] = None
    embed_max_chars: Optional[int] = None
    embed_query_max_chars: Optional[int] = None
    llm_timeout_seconds: Optional[int] = None
    llm_temperature: Optional[float] = None
    llm_max_tokens: Optional[int] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    question: str

# ── Settings endpoints ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """
    Warm the BM25 corpus cache in the background so the first user query
    does not pay the cold-start cost (loading all chunks from ChromaDB).
    For a 77k-chunk index this saves ~4 seconds on the first query.
    """
    import asyncio, threading
    def _warm():
        try:
            from retriever.hybrid_search import _collection, _get_cached_corpus, _get_cached_bm25
            col = _collection()
            if col.count() > 0:
                corpus = _get_cached_corpus(col, None, None)
                _get_cached_bm25(col, None, None)
                print(f"[startup] BM25 corpus warmed: {len(corpus)} chunks cached")
        except Exception as e:
            print(f"[startup] Cache warm failed (index may not exist yet): {e}")
    threading.Thread(target=_warm, daemon=True, name="cache-warmer").start()


@app.get("/api/settings")
def get_settings():
    import config.settings as s
    return {
        "model": s.MODEL,
        "embed_model": s.EMBED_MODEL,
        "ollama_base_url": s.OLLAMA_BASE_URL,
        "max_context_tokens": s.MAX_CONTEXT_TOKENS,
        "top_k_chunks": s.TOP_K_CHUNKS,
        "chunk_max_lines": s.CHUNK_MAX_LINES,
        "embed_workers": s.EMBED_WORKERS,
        "chroma_batch_size": s.CHROMA_BATCH_SIZE,
        "embed_num_ctx": s.EMBED_NUM_CTX,
        "embed_max_chars": s.EMBED_MAX_CHARS,
        "embed_query_max_chars": s.EMBED_QUERY_MAX_CHARS,
        "llm_timeout_seconds": s.LLM_TIMEOUT_SECONDS,
        "llm_temperature": s.LLM_TEMPERATURE,
        "llm_max_tokens": s.LLM_MAX_TOKENS,
    }

@app.post("/api/settings")
def update_settings(req: SettingsUpdate):
    """Persist settings to .env and reload the module."""
    env_path = Path(__file__).parent / ".env"
    env: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    mapping = {
        "model": "CODING_MODEL", "embed_model": "EMBED_MODEL",
        "ollama_base_url": "OLLAMA_BASE_URL",
        "max_context_tokens": "MAX_CONTEXT_TOKENS", "top_k_chunks": "TOP_K_CHUNKS",
        "chunk_max_lines": "CHUNK_MAX_LINES", "embed_workers": "EMBED_WORKERS",
        "chroma_batch_size": "CHROMA_BATCH_SIZE",
        "embed_num_ctx": "EMBED_NUM_CTX",
        "embed_max_chars": "EMBED_MAX_CHARS",
        "embed_query_max_chars": "EMBED_QUERY_MAX_CHARS",
        "llm_timeout_seconds": "LLM_TIMEOUT_SECONDS",
        "llm_temperature": "LLM_TEMPERATURE", "llm_max_tokens": "LLM_MAX_TOKENS",
    }
    for field, env_key in mapping.items():
        val = getattr(req, field)
        if val is not None:
            env[env_key] = str(val)
            os.environ[env_key] = str(val)
    env_path.write_text("\n".join(f"{k}={v}" for k, v in env.items()))
    # Reload settings module so changes take effect immediately
    import importlib, config.settings as s
    importlib.reload(s)
    return {"status": "ok", "saved": list(mapping.keys())}

# ── Status / health ───────────────────────────────────────────────────────────
@app.get("/api/status")
def status():
    import requests as req
    import config.settings as s
    ollama_ok = False
    model_ok = False
    try:
        r = req.get(f"{s.OLLAMA_BASE_URL}/api/tags", timeout=3)
        ollama_ok = True
        models = [m["name"] for m in r.json().get("models", [])]
        model_ok = any(s.MODEL in m for m in models)
    except Exception:
        pass
    from indexer.embedder import get_collection_stats
    stats = get_collection_stats()
    return {
        "ollama_running": ollama_ok,
        "model_available": model_ok,
        "model": s.MODEL,
        "index_chunks": stats["total_chunks"],
        "index_status": stats["status"],
    }

# ── Index (SSE streaming progress) ───────────────────────────────────────────
@app.post("/api/index/stream")
async def index_stream(req: IndexRequest):
    """Stream indexing progress as Server-Sent Events."""
    async def generate() -> AsyncIterator[str]:
        def send(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        try:
            from indexer.strategy import analyze_repo
            from indexer.scanner import scan_repo, repo_id_for_path
            from indexer.chunker import chunk_file
            from indexer.embedder import index_chunks, delete_chunks_for_files, delete_chunks_for_repo
            from indexer.strategy import get_profile

            repo_path = Path(req.path).resolve()
            if not repo_path.exists():
                yield send("error", {"message": f"Path not found: {repo_path}"})
                return

            yield send("status", {"message": f"Analysing {repo_path.name}..."})
            await asyncio.sleep(0)

            if req.auto:
                analysis = analyze_repo(repo_path)
                areas = analysis.areas
                yield send("analysis", {
                    "repo_type": analysis.repo_type,
                    "areas": [{"path": str(a.path), "profile": a.profile,
                                "files": a.estimated_files} for a in areas],
                    "markers": analysis.markers,
                })
            else:
                profile = get_profile(req.profile)
                from indexer.strategy import PlanArea
                areas = [PlanArea(path=repo_path, profile=profile.name, reason="manual")]

            total_indexed = 0
            for area_idx, area in enumerate(areas):
                yield send("status", {"message": f"Scanning {area.path.name} ({area.profile})..."})
                await asyncio.sleep(0)

                repo_id = repo_id_for_path(area.path)
                profile = get_profile(area.profile)
                files, deleted = scan_repo(
                    area.path, incremental=True, force_reindex=req.force,
                    include_extensions=profile.include_extensions,
                    include_filenames=profile.include_filenames,
                    extra_skip_dirs=profile.extra_skip_dirs,
                    max_file_size_kb=profile.max_file_size_kb,
                )

                yield send("scan_done", {"files": len(files), "deleted": len(deleted)})
                await asyncio.sleep(0)

                if req.force:
                    delete_chunks_for_repo(repo_id)
                if deleted:
                    delete_chunks_for_files(deleted, repo_id=repo_id)

                if not files:
                    yield send("status", {"message": "Nothing changed — index is up to date."})
                    continue

                yield send("status", {"message": f"Chunking {len(files)} files..."})
                await asyncio.sleep(0)

                all_chunks = []
                for f in files:
                    try:
                        content = f.path.read_text(encoding="utf-8", errors="replace")
                        all_chunks.extend(chunk_file(f.relative_path, f.language, content, repo_id=repo_id))
                    except Exception:
                        pass

                yield send("chunk_done", {"chunks": len(all_chunks)})
                await asyncio.sleep(0)

                yield send("status", {"message": f"Checking resume state..."})
                await asyncio.sleep(0)
                yield send("status", {"message": f"Embedding {len(all_chunks)} chunks (parallel, resume-safe)..."})

                # Run blocking index_chunks in a thread so we don't block the event loop
                progress_events = []
                def _on_progress(indexed, total, eta):
                    progress_events.append({"indexed": indexed, "total": total, "eta": eta})

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, lambda: index_chunks(all_chunks, show_progress=False,
                                               on_progress=_on_progress, resume=True)
                )
                for ev in progress_events:
                    yield send("index_progress", ev)

                total_indexed += result
                yield send("area_done", {"area": str(area.path.name), "indexed": result})
                await asyncio.sleep(0)

            yield send("done", {"total_indexed": total_indexed})

        except Exception as e:
            yield send("error", {"message": str(e)})

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Ask ───────────────────────────────────────────────────────────────────────
@app.post("/api/ask/stream")
async def ask_stream(req: QueryRequest):
    async def generate():
        if req.use_rag:
            from retriever.rag import rag_stream, understand_query
            from retriever.hybrid_search import retrieve

            step_events = []
            def on_step(name, data):
                step_events.append((name, data))

            # Run query understanding sync first (fast, ~1s)
            loop = asyncio.get_event_loop()
            raq = await loop.run_in_executor(
                None, lambda: understand_query(req.question)
            )
            yield f"data: {json.dumps({'type':'plan','plan':{'search_terms':raq.plan.search_terms,'semantic_query':raq.reformulated,'file_hints':raq.plan.file_hints,'task':raq.plan.task,'reasoning':raq.plan.reasoning,'sub_queries':raq.sub_queries,'hypothetical':raq.hypothetical_answer[:120] if raq.hypothetical_answer else '','query_variants':raq.query_variants or []}})}\n\n"

            # Stream tokens from RAG pipeline
            full = ""
            for token in rag_stream(
                question=req.question,
                top_k=req.top_k,
                repo_root=req.repo_root,
                language_filter=req.language_filter,
                use_compression=req.use_compression,
                check_faithfulness_flag=req.check_faithfulness,
                on_step=on_step,
            ):
                if isinstance(token, str) and token.startswith("__llm_stats__"):
                    continue  # already handled via on_step callback
                full += token
                yield f"data: {json.dumps({'type':'token','text':token})}\n\n"

            # Emit step data collected during generation
            for name, data in step_events:
                if name == "retrieved":
                    yield f"data: {json.dumps({'type':'retrieved','count':len(data),'chunks':[{'file':c.file_path.split('/')[-1],'lines':f'{c.start_line}-{c.end_line}','score':round(c.score,3),'source':'grep' if c.chunk_type=='grep_match' else 'rag'} for c in data]})}\n\n"
                elif name == "context_built":
                    ctx, srcs, tok = data
                    yield f"data: {json.dumps({'type':'sources','files':srcs})}\n\n"
                    yield f"data: {json.dumps({'type':'context_tokens','tokens':tok})}\n\n"
                elif name == "faithfulness":
                    yield f"data: {json.dumps({'type':'faithfulness','result':data})}\n\n"
                elif name == "llm_stats":
                    yield f"data: {json.dumps({'type':'llm_stats','stats':data})}\n\n"

        else:
            from retriever.hybrid_search import retrieve
            from retriever.context_builder import build_context
            from assistant.prompts import build_prompt, build_no_context_prompt
            from assistant.llm import stream_response

            chunks = retrieve(req.question, top_k=req.top_k, language_filter=req.language_filter)
            if chunks:
                context, sources = build_context(chunks, req.question)
                system, user_msg = build_prompt("general", req.question, context, sources)
            else:
                system, user_msg = build_no_context_prompt("general", req.question)
                sources = []
            yield f"data: {json.dumps({'type':'sources','files':sources})}\n\n"
            for token in stream_response(system, user_msg):
                if isinstance(token, str) and token.startswith("__llm_stats__"):
                    try:
                        stats = json.loads(token[len("__llm_stats__"):-len("__end_stats__")])
                        yield f"data: {json.dumps({'type':'llm_stats','stats':stats})}\n\n"
                    except Exception:
                        pass
                    continue
                yield f"data: {json.dumps({'type':'token','text':token})}\n\n"

        yield f"data: {json.dumps({'type':'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})

# ── Review ────────────────────────────────────────────────────────────────────
@app.post("/api/review/stream")
async def review_stream(req: ReviewRequest):
    async def generate():
        from retriever.hybrid_search import retrieve
        from retriever.context_builder import build_context
        from assistant.prompts import build_prompt
        from assistant.llm import stream_response

        query = f"Review this file for bugs, security issues, and improvements: {req.file_path}"
        chunks = retrieve(query, file_filter=req.file_path, top_k=8)
        if not chunks:
            try:
                content = Path(req.file_path).read_text(encoding="utf-8", errors="replace")[:8000]
                context = f"```\n{content}\n```"
                sources = [req.file_path]
            except Exception as e:
                yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"; return
        else:
            context, sources = build_context(chunks, query, "review")
        system, user_msg = build_prompt("review", query, context, sources)
        yield f"data: {json.dumps({'type':'sources','files':sources})}\n\n"
        for token in stream_response(system, user_msg):
            if isinstance(token, str) and token.startswith("__llm_stats__"):
                try:
                    stats = json.loads(token[len("__llm_stats__"):-len("__end_stats__")])
                    yield f"data: {json.dumps({'type':'llm_stats','stats':stats})}\n\n"
                except Exception:
                    pass
                continue
            yield f"data: {json.dumps({'type':'token','text':token})}\n\n"
        yield f"data: {json.dumps({'type':'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})

# ── Explain ───────────────────────────────────────────────────────────────────
@app.post("/api/explain/stream")
async def explain_stream(req: ExplainRequest):
    async def generate():
        from retriever.hybrid_search import retrieve
        from retriever.context_builder import build_context
        from assistant.prompts import build_prompt
        from assistant.llm import stream_response

        if req.function_name:
            query = f"Explain the function `{req.function_name}` in {req.file_path or 'the codebase'}"
        else:
            query = f"Explain what {req.file_path} does and how it works"
        chunks = retrieve(query, file_filter=req.file_path, top_k=5)
        context, sources = build_context(chunks, query) if chunks else ("", [])
        system, user_msg = build_prompt("explain", query, context, sources)
        yield f"data: {json.dumps({'type':'sources','files':sources})}\n\n"
        for token in stream_response(system, user_msg):
            if isinstance(token, str) and token.startswith("__llm_stats__"):
                try:
                    stats = json.loads(token[len("__llm_stats__"):-len("__end_stats__")])
                    yield f"data: {json.dumps({'type':'llm_stats','stats':stats})}\n\n"
                except Exception:
                    pass
                continue
            yield f"data: {json.dumps({'type':'token','text':token})}\n\n"
        yield f"data: {json.dumps({'type':'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})

# ── Generate ──────────────────────────────────────────────────────────────────
@app.post("/api/generate/stream")
async def generate_stream(req: GenerateRequest):
    async def generate():
        from retriever.hybrid_search import retrieve
        from retriever.context_builder import build_context
        from assistant.prompts import build_prompt, build_no_context_prompt
        from assistant.llm import stream_response

        chunks = retrieve(req.description, top_k=4)
        if chunks:
            context, sources = build_context(chunks, req.description, "generate")
            system, user_msg = build_prompt("generate", req.description, context, sources,
                                            extra_instruction="Match the coding style from the context.")
        else:
            system, user_msg = build_no_context_prompt("generate", req.description)
            sources = []
        yield f"data: {json.dumps({'type':'sources','files':sources})}\n\n"
        for token in stream_response(system, user_msg):
            if isinstance(token, str) and token.startswith("__llm_stats__"):
                try:
                    stats = json.loads(token[len("__llm_stats__"):-len("__end_stats__")])
                    yield f"data: {json.dumps({'type':'llm_stats','stats':stats})}\n\n"
                except Exception:
                    pass
                continue
            yield f"data: {json.dumps({'type':'token','text':token})}\n\n"
        yield f"data: {json.dumps({'type':'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})

# ── Chat ──────────────────────────────────────────────────────────────────────
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    async def generate():
        from retriever.hybrid_search import retrieve
        from retriever.context_builder import build_context
        from assistant.llm import stream_response

        chunks = retrieve(req.question, top_k=4)
        context, sources = build_context(chunks, req.question) if chunks else ("", [])
        system = "You are a helpful coding assistant. Answer questions about the codebase accurately and concisely."
        user_msg = f"{context}\n\n---\n\n{req.question}" if context else req.question
        history = [{"role": m.role, "content": m.content} for m in req.messages[-12:]]

        yield f"data: {json.dumps({'type':'sources','files':sources})}\n\n"
        for token in stream_response(system, user_msg, history=history):
            if isinstance(token, str) and token.startswith("__llm_stats__"):
                try:
                    stats = json.loads(token[len("__llm_stats__"):-len("__end_stats__")])
                    yield f"data: {json.dumps({'type':'llm_stats','stats':stats})}\n\n"
                except Exception:
                    pass
                continue
            yield f"data: {json.dumps({'type':'token','text':token})}\n\n"
        yield f"data: {json.dumps({'type':'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})

# ── Serve UI ──────────────────────────────────────────────────────────────────
@app.post("/api/index/stop")
async def index_stop():
    """Signal the indexing worker to stop gracefully."""
    try:
        from indexer.embedder import request_cancel
        request_cancel()
        return {"status": "stop requested"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    ui = Path(__file__).parent / "ui" / "index.html"
    return HTMLResponse(ui.read_text())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
