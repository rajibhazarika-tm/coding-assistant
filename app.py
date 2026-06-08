"""
app.py — Coding Assistant desktop app built with CustomTkinter.

Pure Python UI — no browser, no Electron, no webview.
Runs on Windows/macOS/Linux. Packages to .exe with PyInstaller.

Layout:
┌─────────────────────────────────────────────────────┐
│  Sidebar          │  Tab area                        │
│  ─ Chat           │  [Chat|Ask|Index|Review|Explain  │
│  ─ Ask            │   Generate|Settings]             │
│  ─ Index          │                                  │
│  ─ Review         │  Content panel (scrollable)      │
│  ─ Explain        │                                  │
│  ─ Generate       │  Input bar (context-sensitive)   │
│  ─ Settings       │                                  │
│  ──────────────── │                                  │
│  Status           │                                  │
└─────────────────────────────────────────────────────┘

Run:
    python app.py

Build to .exe (Windows, run in project root with venv active):
    pip install customtkinter pillow pyinstaller
    pyinstaller --onefile --windowed --name CodingAssistant app.py
"""
from __future__ import annotations

import sys
import threading
import queue
import time
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from tkinter import filedialog, messagebox

# ── Appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Colour palette (matches web UI)
C = {
    "bg":       "#0f1117",
    "bg2":      "#1a1d27",
    "bg3":      "#232636",
    "bg4":      "#2d3148",
    "border":   "#2e3250",
    "text":     "#e2e8f0",
    "text2":    "#94a3b8",
    "text3":    "#64748b",
    "accent":   "#6366f1",
    "accent2":  "#818cf8",
    "green":    "#10b981",
    "red":      "#ef4444",
    "yellow":   "#f59e0b",
}

FONT_MONO  = ("Consolas", 12)
FONT_UI    = ("Segoe UI", 13)
FONT_SMALL = ("Segoe UI", 11)
FONT_TITLE = ("Segoe UI Semibold", 14)

NAV_ITEMS = ["💬  Chat", "🔍  Ask", "📦  Index", "🔬  Review",
             "📖  Explain", "⚙️   Generate", "⚙️   Settings"]


# ── Worker thread helper ──────────────────────────────────────────────────────
def run_in_thread(fn, *args, callback=None, error_callback=None):
    """Run fn(*args) in a daemon thread; deliver result via callback on main thread."""
    def _run():
        try:
            result = fn(*args)
            if callback:
                callback(result)
        except Exception as exc:
            if error_callback:
                error_callback(exc)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


# ── Main application ──────────────────────────────────────────────────────────
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Coding Assistant")
        self.geometry("1280x820")
        self.minsize(900, 600)
        self.configure(fg_color=C["bg"])

        # State
        self.chat_history: list[dict] = []
        self._stream_queue: queue.Queue = queue.Queue()
        self._indexing = False

        self._build_layout()
        self._show_panel("chat")
        self.after(500, self._poll_status)
        self.after(100, self._drain_stream_queue)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_layout(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self._sidebar = ctk.CTkFrame(self, width=200, fg_color=C["bg2"],
                                     corner_radius=0, border_width=0)
        self._sidebar.grid(row=0, column=0, sticky="nsew")
        self._sidebar.grid_propagate(False)
        self._sidebar.grid_rowconfigure(len(NAV_ITEMS) + 1, weight=1)

        logo = ctk.CTkLabel(self._sidebar, text="🤖  Coding Assistant",
                            font=ctk.CTkFont("Segoe UI Semibold", 15),
                            text_color=C["accent2"])
        logo.grid(row=0, column=0, padx=16, pady=(18, 12), sticky="w")

        self._nav_buttons: dict[str, ctk.CTkButton] = {}
        for i, label in enumerate(NAV_ITEMS):
            key = label.split()[-1].lower()
            btn = ctk.CTkButton(
                self._sidebar, text=label, anchor="w",
                font=ctk.CTkFont("Segoe UI", 13),
                fg_color="transparent", hover_color=C["bg3"],
                text_color=C["text2"], corner_radius=6,
                command=lambda k=key: self._show_panel(k),
            )
            btn.grid(row=i + 1, column=0, padx=8, pady=2, sticky="ew")
            self._nav_buttons[key] = btn

        # Status area at bottom of sidebar
        self._status_frame = ctk.CTkFrame(self._sidebar, fg_color=C["bg3"],
                                           corner_radius=6)
        self._status_frame.grid(row=len(NAV_ITEMS) + 2, column=0,
                                 padx=8, pady=(0, 10), sticky="sew")
        self._lbl_ollama = ctk.CTkLabel(self._status_frame, text="⏳ Checking Ollama…",
                                         font=ctk.CTkFont("Segoe UI", 11),
                                         text_color=C["text3"])
        self._lbl_ollama.grid(row=0, padx=10, pady=(8, 2), sticky="w")
        self._lbl_chunks = ctk.CTkLabel(self._status_frame, text="",
                                         font=ctk.CTkFont("Segoe UI", 11),
                                         text_color=C["text3"])
        self._lbl_chunks.grid(row=1, padx=10, pady=(0, 8), sticky="w")

        # Main panel container
        self._main = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        self._main.grid(row=0, column=1, sticky="nsew")
        self._main.grid_columnconfigure(0, weight=1)
        self._main.grid_rowconfigure(0, weight=1)

        self._panels: dict[str, ctk.CTkFrame] = {}
        self._build_chat_panel()
        self._build_ask_panel()
        self._build_index_panel()
        self._build_review_panel()
        self._build_explain_panel()
        self._build_generate_panel()
        self._build_settings_panel()

    def _show_panel(self, key: str):
        for k, p in self._panels.items():
            p.grid_remove()
        if key in self._panels:
            self._panels[key].grid(row=0, column=0, sticky="nsew")
        for k, btn in self._nav_buttons.items():
            btn.configure(
                fg_color=C["bg4"] if k == key else "transparent",
                text_color=C["accent2"] if k == key else C["text2"],
            )

    def _make_panel(self, key: str) -> ctk.CTkFrame:
        p = ctk.CTkFrame(self._main, fg_color=C["bg"], corner_radius=0)
        p.grid_columnconfigure(0, weight=1)
        self._panels[key] = p
        return p

    # ── Chat panel ────────────────────────────────────────────────────────────
    def _build_chat_panel(self):
        p = self._make_panel("chat")
        p.grid_rowconfigure(0, weight=1)

        self._chat_scroll = ctk.CTkScrollableFrame(p, fg_color=C["bg"],
                                                    corner_radius=0)
        self._chat_scroll.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self._chat_scroll.grid_columnconfigure(0, weight=1)
        self._chat_row = 0

        # Input bar
        bar = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=0, height=60)
        bar.grid(row=1, column=0, sticky="ew")
        bar.grid_columnconfigure(0, weight=1)
        bar.grid_propagate(False)

        self._chat_input = ctk.CTkTextbox(bar, height=40, font=ctk.CTkFont("Segoe UI", 13),
                                           fg_color=C["bg3"], border_color=C["border"],
                                           border_width=1, corner_radius=6,
                                           text_color=C["text"])
        self._chat_input.grid(row=0, column=0, padx=(12, 6), pady=10, sticky="ew")
        self._chat_input.bind("<Return>", self._chat_enter)
        self._chat_input.bind("<Shift-Return>", lambda e: None)

        ctk.CTkButton(bar, text="Send", width=80,
                      font=ctk.CTkFont("Segoe UI Semibold", 13),
                      fg_color=C["accent"], hover_color=C["accent2"],
                      command=self._send_chat).grid(row=0, column=1, padx=(0, 12), pady=10)

        # Welcome bubble
        self._add_chat_bubble("assistant",
            "Hello! Index a repo first, then ask me anything — flows, errors, "
            "explanations, code generation. I use a 5-step pipeline to find "
            "the most relevant code before answering.")

    def _chat_enter(self, event):
        if not event.state & 0x1:  # Shift not held
            self._send_chat()
            return "break"

    def _add_chat_bubble(self, role: str, text: str) -> ctk.CTkTextbox:
        is_user = role == "user"
        outer = ctk.CTkFrame(self._chat_scroll,
                             fg_color=C["accent"] if is_user else C["bg3"],
                             corner_radius=10)
        pad = (120, 12) if is_user else (12, 120)
        outer.grid(row=self._chat_row, column=0, padx=pad, pady=(4, 4), sticky="e" if is_user else "w")
        self._chat_row += 1

        lbl = ctk.CTkTextbox(outer, font=ctk.CTkFont("Segoe UI", 13),
                              fg_color="transparent", text_color="#ffffff" if is_user else C["text"],
                              wrap="word", height=60, activate_scrollbars=False,
                              border_width=0)
        lbl.grid(padx=12, pady=8, sticky="ew")
        lbl.insert("1.0", text)
        lbl.configure(state="disabled")
        self._resize_textbox(lbl, text)
        # Scroll to bottom
        self.after(50, lambda: self._chat_scroll._parent_canvas.yview_moveto(1.0))
        return lbl

    def _resize_textbox(self, tb: ctk.CTkTextbox, text: str):
        lines = max(1, text.count("\n") + 1, len(text) // 70 + 1)
        tb.configure(height=min(lines * 20 + 16, 400))

    def _send_chat(self):
        q = self._chat_input.get("1.0", "end").strip()
        if not q:
            return
        self._chat_input.delete("1.0", "end")
        self._add_chat_bubble("user", q)
        reply_box = self._add_chat_bubble("assistant", "⏳ Thinking…")

        def _stream():
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from retriever.pipeline import run_pipeline, PipelineResult
                from assistant.prompts import build_prompt
                from assistant.llm import stream_response
                import config.settings as s

                result: PipelineResult = run_pipeline(q, top_k=5)
                system, user_msg = build_prompt(
                    result.plan.task, q, result.context, result.sources)
                history = self.chat_history[-12:]

                full = ""
                for token in stream_response(system, user_msg, history=history):
                    full += token
                    if len(full) % 60 == 0:
                        self._stream_queue.put(("chat_token", reply_box, full))
                self._stream_queue.put(("chat_token", reply_box, full))
                self._stream_queue.put(("chat_done", reply_box, full))
                self.chat_history.append({"role": "user", "content": q})
                self.chat_history.append({"role": "assistant", "content": full})
            except Exception as e:
                self._stream_queue.put(("chat_token", reply_box, f"❌ Error: {e}"))

        threading.Thread(target=_stream, daemon=True).start()

    # ── Ask panel ─────────────────────────────────────────────────────────────
    def _build_ask_panel(self):
        p = self._make_panel("ask")
        # row 0 = controls (fixed), row 1 = trace (fixed, collapsible),
        # row 2 = output (expands to fill all remaining space)
        p.grid_rowconfigure(2, weight=1)
        p.grid_rowconfigure(0, weight=0)
        p.grid_rowconfigure(1, weight=0)

        # Controls
        ctrl = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=8)
        ctrl.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        ctrl.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(ctrl, text="Ask a question about your codebase",
                     font=ctk.CTkFont("Segoe UI Semibold", 13),
                     text_color=C["text2"]).grid(row=0, column=0, padx=14, pady=(12, 4), sticky="w")

        self._ask_input = ctk.CTkTextbox(ctrl, height=80, font=ctk.CTkFont("Segoe UI", 13),
                                          fg_color=C["bg3"], border_color=C["border"],
                                          border_width=1, corner_radius=6, text_color=C["text"])
        self._ask_input.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="ew")
        self._ask_input.insert("1.0", "e.g. How does authentication work?\n"
                                       "What causes the NullPointerException at OrderService:87?")

        opts = ctk.CTkFrame(ctrl, fg_color="transparent")
        opts.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="ew")
        opts.grid_columnconfigure(1, weight=1)

        self._ask_repo = ctk.CTkEntry(opts, placeholder_text="Repo path for grep (optional)",
                                       font=ctk.CTkFont("Segoe UI", 12),
                                       fg_color=C["bg3"], border_color=C["border"],
                                       text_color=C["text"])
        self._ask_repo.grid(row=0, column=0, padx=(0, 8), sticky="ew")
        ctk.CTkButton(opts, text="Browse", width=70,
                      fg_color=C["bg4"], hover_color=C["bg3"], text_color=C["text2"],
                      command=lambda: self._browse_dir(self._ask_repo)).grid(row=0, column=1, padx=(0, 8))

        self._ask_topk = ctk.CTkEntry(opts, width=50, placeholder_text="5",
                                       font=ctk.CTkFont("Segoe UI", 12),
                                       fg_color=C["bg3"], border_color=C["border"],
                                       text_color=C["text"])
        self._ask_topk.grid(row=0, column=2, padx=(0, 8))
        ctk.CTkLabel(opts, text="top-k", font=ctk.CTkFont("Segoe UI", 11),
                     text_color=C["text3"]).grid(row=0, column=3, padx=(0, 16))

        self._ask_btn = ctk.CTkButton(opts, text="Ask", width=90,
                                       fg_color=C["accent"], hover_color=C["accent2"],
                                       font=ctk.CTkFont("Segoe UI Semibold", 13),
                                       command=self._run_ask)
        self._ask_btn.grid(row=0, column=4)

        # Pipeline trace
        self._ask_trace = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=8)
        self._ask_trace.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="ew")
        self._ask_trace.grid_columnconfigure(0, weight=1)
        self._ask_trace_lbl = ctk.CTkLabel(self._ask_trace,
                     text="🔍 Retrieval pipeline  ▶ show",
                     font=ctk.CTkFont("Segoe UI Semibold", 12),
                     text_color=C["text2"], cursor="hand2")
        self._ask_trace_lbl.grid(row=0, column=0, padx=14, pady=(8, 4), sticky="w")
        self._ask_trace_lbl.bind("<Button-1>", lambda e: self._toggle_ask_trace())
        self._ask_trace_text = ctk.CTkTextbox(self._ask_trace, height=110,
                                               font=ctk.CTkFont("Consolas", 11),
                                               fg_color=C["bg"], border_width=0,
                                               text_color=C["text2"])
        self._ask_trace_text.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="ew")
        self._ask_trace_text.insert("1.0", "Run a query to see the pipeline steps here.")
        self._ask_trace_text.configure(state="disabled")
        # Start collapsed — user can click the title to expand
        self._ask_trace_collapsed = True
        self._ask_trace_text.grid_remove()

        # Output
        out_frame = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=8)
        out_frame.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="nsew")
        out_frame.grid_columnconfigure(0, weight=1)
        out_frame.grid_rowconfigure(1, weight=1)
        self._ask_sources_lbl = ctk.CTkLabel(out_frame, text="",
                                              font=ctk.CTkFont("Consolas", 11),
                                              text_color=C["accent2"])
        self._ask_sources_lbl.grid(row=0, column=0, padx=14, pady=(8, 0), sticky="w")
        self._ask_output = ctk.CTkTextbox(out_frame,
                                           font=ctk.CTkFont("Segoe UI", 13),
                                           fg_color=C["bg3"], border_width=0,
                                           text_color=C["text"], wrap="word",
                                           height=400)   # min height so it's never tiny
        self._ask_output.grid(row=1, column=0, padx=12, pady=(4, 12), sticky="nsew")
        self._ask_output.insert("1.0", "Answer will appear here.")
        self._ask_output.configure(state="disabled")

    def _toggle_ask_trace(self):
        if self._ask_trace_collapsed:
            self._ask_trace_text.grid()
            self._ask_trace_lbl.configure(text="🔍 Retrieval pipeline  ▼ hide")
            self._ask_trace_collapsed = False
        else:
            self._ask_trace_text.grid_remove()
            self._ask_trace_lbl.configure(text="🔍 Retrieval pipeline  ▶ show")
            self._ask_trace_collapsed = True

    def _run_ask(self):
        q = self._ask_input.get("1.0", "end").strip()
        if not q:
            return
        repo = self._ask_repo.get().strip() or None
        try:
            top_k = int(self._ask_topk.get().strip() or "5")
        except ValueError:
            top_k = 5

        self._ask_btn.configure(state="disabled", text="⏳ Searching…")
        self._set_output(self._ask_output, "⏳ Running pipeline…")
        # Auto-expand trace when query starts
        self._ask_trace_text.grid()
        self._ask_trace_lbl.configure(text="🔍 Retrieval pipeline  ▼ hide")
        self._ask_trace_collapsed = False
        self._set_output(self._ask_trace_text, "Step 1/5 — Analysing query…")
        self._ask_sources_lbl.configure(text="")

        def _work():
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from retriever.rag import rag_stream, understand_query

                trace_lines = []

                def on_step(name, data):
                    if name == "query_understood":
                        raq = data
                        lines = [f"① Query understanding"]
                        # Show corrections if any were made
                        if raq.corrections:
                            lines.append(f"   Corrections: {'; '.join(raq.corrections[:4])}")
                        if raq.corrected != raq.original:
                            lines.append(f"   Corrected: {raq.corrected[:80]}")
                        lines += [
                            f"   Task: {raq.plan.task}  multi-hop: {raq.is_multi_hop}",
                            f"   Reformulated: {raq.reformulated[:80]}",
                            f"   Grep terms: {raq.plan.search_terms}",
                            (f"   Hypothetical: {raq.hypothetical_answer[:80]}...") if raq.hypothetical_answer else "",
                        ]
                        trace_lines.extend(l for l in lines if l)
                        self._stream_queue.put(("ask_trace", "\n".join(trace_lines)))

                    elif name == "retrieved":
                        chunks = data
                        trace_lines.append(f"② Retrieved {len(chunks)} chunks (HDE + vector + BM25 + grep)")
                        for c in chunks[:5]:
                            src = "grep" if c.chunk_type == "grep_match" else "rag"
                            trace_lines.append(f"   [{src}] {c.file_path.split('/')[-1]}:{c.start_line}-{c.end_line} score={c.score:.3f}")
                        self._stream_queue.put(("ask_trace", "\n".join(trace_lines)))

                    elif name == "compressed":
                        trace_lines.append(f"③ Contextual compression applied")
                        self._stream_queue.put(("ask_trace", "\n".join(trace_lines)))

                    elif name == "context_built":
                        ctx, srcs, tok = data
                        trace_lines.append(f"④ Context: ~{tok} tokens from {len(srcs)} files")
                        self._stream_queue.put(("ask_trace", "\n".join(trace_lines)))
                        if srcs:
                            self._stream_queue.put(("ask_sources",
                                "Sources: " + "  ·  ".join(Path(s).name for s in srcs)))

                    elif name == "generating":
                        trace_lines.append(f"⑤ Generating answer…")
                        self._stream_queue.put(("ask_trace", "\n".join(trace_lines)))
                        self._stream_queue.put(("ask_token", ""))

                    elif name == "faithfulness":
                        f = data
                        verdict = "✅ Faithful" if f.get("is_faithful") else "⚠️ May hallucinate"
                        trace_lines.append(f"⑥ Faithfulness: {verdict} — {f.get('verdict','')}")
                        self._stream_queue.put(("ask_trace", "\n".join(trace_lines)))

                full = ""
                for token in rag_stream(
                    question=q, top_k=top_k,
                    repo_root=repo or None,
                    on_step=on_step,
                ):
                    full += token
                    if len(full) % 60 == 0:
                        self._stream_queue.put(("ask_token", full))
                self._stream_queue.put(("ask_token", full))
                self._stream_queue.put(("ask_done", full))
            except Exception as e:
                self._stream_queue.put(("ask_token", f"❌ Error: {e}"))
                self._stream_queue.put(("ask_done", ""))

        threading.Thread(target=_work, daemon=True).start()

    # ── Index panel ───────────────────────────────────────────────────────────
    def _build_index_panel(self):
        p = self._make_panel("index")
        p.grid_rowconfigure(2, weight=1)

        ctrl = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=8)
        ctrl.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        ctrl.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(ctrl, text="Repository path",
                     font=ctk.CTkFont("Segoe UI", 12),
                     text_color=C["text2"]).grid(row=0, column=0, padx=(14, 8), pady=12)
        self._index_path = ctk.CTkEntry(ctrl, font=ctk.CTkFont("Segoe UI", 13),
                                         fg_color=C["bg3"], border_color=C["border"],
                                         text_color=C["text"],
                                         placeholder_text="C:\\projects\\my-repo")
        self._index_path.grid(row=0, column=1, padx=(0, 8), pady=12, sticky="ew")
        ctk.CTkButton(ctrl, text="Browse", width=80,
                      fg_color=C["bg4"], hover_color=C["bg3"], text_color=C["text2"],
                      command=lambda: self._browse_dir(self._index_path)).grid(row=0, column=2, padx=(0, 8))

        opts = ctk.CTkFrame(ctrl, fg_color="transparent")
        opts.grid(row=1, column=0, columnspan=3, padx=14, pady=(0, 12), sticky="ew")
        self._index_auto = ctk.CTkCheckBox(opts, text="Auto-detect profile",
                                            font=ctk.CTkFont("Segoe UI", 12),
                                            text_color=C["text2"])
        self._index_auto.select()
        self._index_auto.grid(row=0, column=0, padx=(0, 20))
        self._index_force = ctk.CTkCheckBox(opts, text="Force full re-index",
                                             font=ctk.CTkFont("Segoe UI", 12),
                                             text_color=C["text2"])
        self._index_force.grid(row=0, column=1, padx=(0, 20))
        self._index_btn = ctk.CTkButton(opts, text="▶  Start Indexing",
                                         fg_color=C["accent"], hover_color=C["accent2"],
                                         font=ctk.CTkFont("Segoe UI Semibold", 13),
                                         command=self._run_index)
        self._index_btn.grid(row=0, column=2)
        self._index_stop_btn = ctk.CTkButton(opts, text="⏹  Stop",
                                              fg_color=C["red"], hover_color="#dc2626",
                                              font=ctk.CTkFont("Segoe UI Semibold", 13),
                                              command=self._stop_index, state="disabled")
        self._index_stop_btn.grid(row=0, column=3, padx=(8, 0))

        # Progress
        prog = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=8)
        prog.grid(row=1, column=0, padx=16, pady=(0, 8), sticky="ew")
        prog.grid_columnconfigure(0, weight=1)
        self._index_pbar = ctk.CTkProgressBar(prog, fg_color=C["bg4"],
                                               progress_color=C["accent"])
        self._index_pbar.grid(row=0, column=0, padx=14, pady=(12, 4), sticky="ew")
        self._index_pbar.set(0)
        self._index_plabel = ctk.CTkLabel(prog, text="Ready",
                                           font=ctk.CTkFont("Segoe UI", 12),
                                           text_color=C["text2"])
        self._index_plabel.grid(row=1, column=0, padx=14, pady=(0, 10), sticky="w")

        # Log
        log_frame = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=8)
        log_frame.grid(row=2, column=0, padx=16, pady=(0, 16), sticky="nsew")
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(log_frame, text="Log", font=ctk.CTkFont("Segoe UI Semibold", 12),
                     text_color=C["text2"]).grid(row=0, column=0, padx=14, pady=(8, 4), sticky="w")
        self._index_log = ctk.CTkTextbox(log_frame, font=FONT_MONO,
                                          fg_color=C["bg"], text_color=C["text2"],
                                          border_width=0)
        self._index_log.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self._index_log.configure(state="disabled")

    def _run_index(self):
        path = self._index_path.get().strip()
        if not path or not Path(path).exists():
            messagebox.showerror("Error", "Please enter a valid repository path.")
            return
        if self._indexing:
            return
        self._indexing = True
        self._index_btn.configure(state="disabled", text="⏳ Indexing…")
        self._index_stop_btn.configure(state="normal")
        self._lbl_chunks.configure(text="📦 Indexing…")
        self._poll_status()  # immediate poll when indexing starts
        self._index_pbar.set(0)
        self._append_log(self._index_log, f"Starting index: {path}\n")

        force = self._index_force.get()
        auto  = self._index_auto.get()

        def _work():
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from indexer.scanner import repo_id_for_path
                from indexer.embedder import request_cancel, reset_cancel
                from indexer.strategy import analyze_repo, get_profile
                from indexer.batch import index_in_batches
                reset_cancel()

                repo_path = Path(path)
                repo_id = repo_id_for_path(repo_path)

                # Auto-detect framework profile
                if auto:
                    analysis = analyze_repo(repo_path)
                    areas = [(a.path, a.profile) for a in analysis.areas]
                    self._stream_queue.put(("index_log",
                        f"Detected: {analysis.repo_type}, {len(analysis.areas)} area(s)\n"))
                else:
                    areas = [(repo_path, "generic")]

                total_indexed = 0
                for area_path, profile_name in areas:
                    profile = get_profile(profile_name)
                    self._stream_queue.put(("index_log",
                        f"Area: {area_path.name}/ [{profile_name}]\n"))

                    def _on_log(msg: str):
                        self._stream_queue.put(("index_log", msg))

                    def _on_progress(bp):
                        # Overall progress bar: batch_index / total_batches
                        overall_pct = bp.batch_index / max(bp.total_batches, 1)
                        label = (
                            f"[{bp.batch_index}/{bp.total_batches}] {bp.folder_name}  "
                            f"{bp.chunks_indexed:,} chunks  ETA {bp.eta_str}"
                        )
                        self._stream_queue.put((
                            "index_progress",
                            overall_pct,
                            label,
                            bp.total_indexed_so_far,
                        ))

                    n = index_in_batches(
                        repo_root=area_path,
                        repo_id=repo_id,
                        profile=profile,
                        force=force,
                        on_progress=_on_progress,
                        on_log=_on_log,
                    )
                    total_indexed += n

                # Invalidate the query correction symbol vocab so it
                # picks up newly indexed names on the next query
                try:
                    from retriever.query_correction import invalidate_vocab_cache
                    invalidate_vocab_cache()
                except Exception:
                    pass
                self._stream_queue.put(("index_done",
                    f"Complete — {total_indexed:,} total chunks indexed"))
            except Exception as e:
                import traceback
                self._stream_queue.put(("index_log",
                    f"❌ Error: {e}\n{traceback.format_exc()}\n"))
                self._stream_queue.put(("index_done", ""))

        threading.Thread(target=_work, daemon=True).start()

    def _stop_index(self):
        """Request cancel — embed workers finish current request then stop."""
        try:
            from indexer.embedder import request_cancel
            request_cancel()
        except Exception:
            pass
        self._index_stop_btn.configure(state="disabled")
        self._append_log(self._index_log, "⚠️  Stop requested — saving progress...\n")

    # ── Review panel ──────────────────────────────────────────────────────────
    def _build_review_panel(self):
        p = self._make_panel("review")
        p.grid_rowconfigure(1, weight=1)
        self._review_output, _ = self._simple_query_panel(
            p, "Review File", "File path",
            "src/OrderService.java",
            "Review", self._run_review,
            browse_file=True,
        )

    def _run_review(self):
        fp = self._review_path.get().strip()
        if not fp:
            return
        self._review_btn.configure(state="disabled", text="⏳")
        self._set_output(self._review_output, "⏳ Reviewing…")
        self._stream_task(
            query=f"Review this file for bugs, security issues, and improvements: {fp}",
            task="review", file_filter=fp,
            output_box=self._review_output,
            btn=self._review_btn, btn_label="Review",
        )

    # ── Explain panel ─────────────────────────────────────────────────────────
    def _build_explain_panel(self):
        p = self._make_panel("explain")
        p.grid_rowconfigure(2, weight=1)

        ctrl = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=8)
        ctrl.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        ctrl.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(ctrl, text="File path (optional)", font=ctk.CTkFont("Segoe UI", 12),
                     text_color=C["text2"]).grid(row=0, column=0, padx=(14, 8), pady=(12, 4))
        self._explain_file = ctk.CTkEntry(ctrl, fg_color=C["bg3"], border_color=C["border"],
                                           text_color=C["text"], font=ctk.CTkFont("Segoe UI", 13),
                                           placeholder_text="src/AuthService.java")
        self._explain_file.grid(row=0, column=1, padx=(0, 8), pady=(12, 4), sticky="ew")
        ctk.CTkButton(ctrl, text="Browse", width=80,
                      fg_color=C["bg4"], hover_color=C["bg3"], text_color=C["text2"],
                      command=lambda: self._browse_file(self._explain_file)).grid(
            row=0, column=2, padx=(0, 12), pady=(12, 4))

        ctk.CTkLabel(ctrl, text="Function name (optional)", font=ctk.CTkFont("Segoe UI", 12),
                     text_color=C["text2"]).grid(row=1, column=0, padx=(14, 8), pady=(4, 12))
        self._explain_fn = ctk.CTkEntry(ctrl, fg_color=C["bg3"], border_color=C["border"],
                                         text_color=C["text"], font=ctk.CTkFont("Segoe UI", 13),
                                         placeholder_text="processOrder")
        self._explain_fn.grid(row=1, column=1, padx=(0, 8), pady=(4, 12), sticky="ew")
        self._explain_btn = ctk.CTkButton(ctrl, text="Explain", width=90,
                                           fg_color=C["accent"], hover_color=C["accent2"],
                                           font=ctk.CTkFont("Segoe UI Semibold", 13),
                                           command=self._run_explain)
        self._explain_btn.grid(row=1, column=2, padx=(0, 12), pady=(4, 12))

        out_frame = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=8)
        out_frame.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nsew")
        out_frame.grid_rowconfigure(0, weight=1)
        out_frame.grid_columnconfigure(0, weight=1)
        self._explain_output = ctk.CTkTextbox(out_frame, font=ctk.CTkFont("Segoe UI", 13),
                                               fg_color=C["bg3"], border_width=0,
                                               text_color=C["text"], wrap="word")
        self._explain_output.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        self._explain_output.insert("1.0", "Explanation will appear here.")
        self._explain_output.configure(state="disabled")

    def _run_explain(self):
        fp = self._explain_file.get().strip() or None
        fn = self._explain_fn.get().strip() or None
        if not fp and not fn:
            messagebox.showerror("Error", "Provide a file path or function name.")
            return
        q = (f"Explain the function `{fn}` in {fp or 'the codebase'}"
             if fn else f"Explain what {fp} does and how it works")
        self._explain_btn.configure(state="disabled", text="⏳")
        self._set_output(self._explain_output, "⏳ Explaining…")
        self._stream_task(query=q, task="explain", file_filter=fp,
                          output_box=self._explain_output,
                          btn=self._explain_btn, btn_label="Explain")

    # ── Generate panel ────────────────────────────────────────────────────────
    def _build_generate_panel(self):
        p = self._make_panel("generate")
        p.grid_rowconfigure(1, weight=1)

        ctrl = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=8)
        ctrl.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        ctrl.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(ctrl, text="Describe what to generate",
                     font=ctk.CTkFont("Segoe UI Semibold", 13),
                     text_color=C["text2"]).grid(row=0, column=0, padx=14, pady=(12, 4), sticky="w")
        self._gen_input = ctk.CTkTextbox(ctrl, height=80, font=ctk.CTkFont("Segoe UI", 13),
                                          fg_color=C["bg3"], border_color=C["border"],
                                          border_width=1, corner_radius=6, text_color=C["text"])
        self._gen_input.grid(row=1, column=0, padx=12, pady=(0, 8), sticky="ew")
        self._gen_input.insert("1.0", "e.g. Add a REST endpoint to list all active users with pagination")
        self._gen_btn = ctk.CTkButton(ctrl, text="Generate", width=100,
                                       fg_color=C["accent"], hover_color=C["accent2"],
                                       font=ctk.CTkFont("Segoe UI Semibold", 13),
                                       command=self._run_generate)
        self._gen_btn.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="e")

        out_frame = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=8)
        out_frame.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nsew")
        out_frame.grid_rowconfigure(0, weight=1)
        out_frame.grid_columnconfigure(0, weight=1)
        self._gen_output = ctk.CTkTextbox(out_frame, font=FONT_MONO,
                                           fg_color=C["bg3"], border_width=0,
                                           text_color=C["text"], wrap="none")
        self._gen_output.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        self._gen_output.insert("1.0", "Generated code will appear here.")
        self._gen_output.configure(state="disabled")

    def _run_generate(self):
        desc = self._gen_input.get("1.0", "end").strip()
        if not desc:
            return
        self._gen_btn.configure(state="disabled", text="⏳")
        self._set_output(self._gen_output, "⏳ Generating…")
        self._stream_task(query=desc, task="generate",
                          output_box=self._gen_output,
                          btn=self._gen_btn, btn_label="Generate",
                          extra_instruction="Match the coding style from the codebase context.")

    # ── Settings panel ────────────────────────────────────────────────────────
    def _build_settings_panel(self):
        p = self._make_panel("settings")
        p.grid_rowconfigure(99, weight=1)

        def _section(row, title):
            f = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=8)
            f.grid(row=row, column=0, padx=16, pady=(8, 0), sticky="ew")
            f.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(f, text=title, font=ctk.CTkFont("Segoe UI Semibold", 12),
                         text_color=C["text2"]).grid(row=0, column=0, columnspan=4,
                                                      padx=14, pady=(10, 6), sticky="w")
            return f

        def _field(frame, row, col, label, var_name, placeholder="", width=220):
            ctk.CTkLabel(frame, text=label, font=ctk.CTkFont("Segoe UI", 12),
                         text_color=C["text2"]).grid(row=row, column=col * 2,
                                                      padx=(14, 8), pady=6, sticky="w")
            e = ctk.CTkEntry(frame, font=ctk.CTkFont("Segoe UI", 12),
                             fg_color=C["bg3"], border_color=C["border"],
                             text_color=C["text"], width=width,
                             placeholder_text=placeholder)
            e.grid(row=row, column=col * 2 + 1, padx=(0, 14), pady=6, sticky="ew")
            setattr(self, f"_s_{var_name}", e)
            return e

        s1 = _section(0, "Model & API")
        s1.grid_columnconfigure(1, weight=1)
        s1.grid_columnconfigure(3, weight=1)
        _field(s1, 1, 0, "LLM Model", "model", "qwen2.5-coder:7b")
        _field(s1, 1, 1, "Embed Model", "embed", "nomic-embed-text")
        _field(s1, 2, 0, "Ollama URL", "url", "http://localhost:11434", width=300)

        s2 = _section(1, "Context & Memory")
        _field(s2, 1, 0, "Max Context Tokens", "ctx", "4096")
        _field(s2, 1, 1, "Top-K Chunks", "topk", "5")
        _field(s2, 2, 0, "Chunk Max Lines", "chunk", "50")
        _field(s2, 2, 1, "LLM Temperature", "temp", "0.1")
        _field(s2, 3, 0, "LLM Max Tokens", "maxtok", "1024")

        s3 = _section(2, "Indexing Performance")
        _field(s3, 1, 0, "Embed Workers", "workers", "2")
        _field(s3, 1, 1, "ChromaDB Batch Size", "batch", "128")
        _field(s3, 2, 0, "Embed num_ctx", "numctx", "8192")
        _field(s3, 2, 1, "Embed max chars/chunk", "maxchars", "2048")
        _field(s3, 3, 0, "Embed max chars/query", "querymaxchars", "1500")

        save_row = ctk.CTkFrame(p, fg_color="transparent")
        save_row.grid(row=3, column=0, padx=16, pady=16, sticky="ew")
        ctk.CTkButton(save_row, text="Save Settings",
                      fg_color=C["accent"], hover_color=C["accent2"],
                      font=ctk.CTkFont("Segoe UI Semibold", 13),
                      command=self._save_settings).pack(side="left", padx=(0, 10))
        ctk.CTkButton(save_row, text="Reload",
                      fg_color=C["bg4"], hover_color=C["bg3"], text_color=C["text2"],
                      command=self._load_settings).pack(side="left")
        self._settings_saved_lbl = ctk.CTkLabel(save_row, text="",
                                                  font=ctk.CTkFont("Segoe UI", 12),
                                                  text_color=C["green"])
        self._settings_saved_lbl.pack(side="left", padx=12)

        self.after(200, self._load_settings)

    def _load_settings(self):
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            import config.settings as s
            import importlib; importlib.reload(s)
            mapping = {
                "model": s.MODEL, "embed": s.EMBED_MODEL,
                "url": s.OLLAMA_BASE_URL,
                "ctx": s.MAX_CONTEXT_TOKENS, "topk": s.TOP_K_CHUNKS,
                "chunk": s.CHUNK_MAX_LINES, "temp": s.LLM_TEMPERATURE,
                "maxtok": s.LLM_MAX_TOKENS,
                "timeout": s.LLM_TIMEOUT_SECONDS,
                "workers": s.EMBED_WORKERS, "batch": s.CHROMA_BATCH_SIZE,
                "numctx": s.EMBED_NUM_CTX,
                "maxchars": s.EMBED_MAX_CHARS,
                "querymaxchars": s.EMBED_QUERY_MAX_CHARS,
            }
            for key, val in mapping.items():
                e = getattr(self, f"_s_{key}", None)
                if e:
                    e.delete(0, "end")
                    e.insert(0, str(val))
        except Exception:
            pass

    def _save_settings(self):
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            import os, importlib, config.settings as s
            env_path = Path(__file__).parent / ".env"
            env: dict = {}
            if env_path.exists():
                for line in env_path.read_text().splitlines():
                    if "=" in line and not line.startswith("#"):
                        k, _, v = line.partition("=")
                        env[k.strip()] = v.strip()
            mapping = {
                "model": "CODING_MODEL", "embed": "EMBED_MODEL",
                "url": "OLLAMA_BASE_URL", "ctx": "MAX_CONTEXT_TOKENS",
                "topk": "TOP_K_CHUNKS", "chunk": "CHUNK_MAX_LINES",
                "temp": "LLM_TEMPERATURE", "maxtok": "LLM_MAX_TOKENS",
                "workers": "EMBED_WORKERS", "batch": "CHROMA_BATCH_SIZE",
                "numctx": "EMBED_NUM_CTX",
            "maxchars": "EMBED_MAX_CHARS",
            "querymaxchars": "EMBED_QUERY_MAX_CHARS",
            }
            for field_key, env_key in mapping.items():
                e = getattr(self, f"_s_{field_key}", None)
                if e:
                    val = e.get().strip()
                    if val:
                        env[env_key] = val
                        os.environ[env_key] = val
            env_path.write_text("\n".join(f"{k}={v}" for k, v in env.items()))
            importlib.reload(s)
            self._settings_saved_lbl.configure(text="✓ Saved")
            self.after(2000, lambda: self._settings_saved_lbl.configure(text=""))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _simple_query_panel(self, p, title, path_label, placeholder,
                            btn_label, cmd, browse_file=False):
        ctrl = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=8)
        ctrl.grid(row=0, column=0, padx=16, pady=(16, 8), sticky="ew")
        ctrl.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(ctrl, text=path_label, font=ctk.CTkFont("Segoe UI", 12),
                     text_color=C["text2"]).grid(row=0, column=0, padx=(14, 8), pady=12)
        entry = ctk.CTkEntry(ctrl, font=ctk.CTkFont("Segoe UI", 13),
                             fg_color=C["bg3"], border_color=C["border"],
                             text_color=C["text"], placeholder_text=placeholder)
        entry.grid(row=0, column=1, padx=(0, 8), pady=12, sticky="ew")
        browse_cmd = (lambda: self._browse_file(entry)) if browse_file else (lambda: self._browse_dir(entry))
        ctk.CTkButton(ctrl, text="Browse", width=80,
                      fg_color=C["bg4"], hover_color=C["bg3"], text_color=C["text2"],
                      command=browse_cmd).grid(row=0, column=2, padx=(0, 8))
        btn = ctk.CTkButton(ctrl, text=btn_label, width=100,
                            fg_color=C["accent"], hover_color=C["accent2"],
                            font=ctk.CTkFont("Segoe UI Semibold", 13),
                            command=cmd)
        btn.grid(row=0, column=3, padx=(0, 12))

        # store references for the specific panel
        if "review" in title.lower():
            self._review_path = entry
            self._review_btn  = btn
        
        out_frame = ctk.CTkFrame(p, fg_color=C["bg2"], corner_radius=8)
        out_frame.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nsew")
        out_frame.grid_rowconfigure(0, weight=1)
        out_frame.grid_columnconfigure(0, weight=1)
        out = ctk.CTkTextbox(out_frame, font=ctk.CTkFont("Segoe UI", 13),
                             fg_color=C["bg3"], border_width=0,
                             text_color=C["text"], wrap="word")
        out.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        out.insert("1.0", "Output will appear here.")
        out.configure(state="disabled")
        return out, entry

    def _stream_task(self, query: str, task: str,
                     output_box: ctk.CTkTextbox,
                     btn: ctk.CTkButton, btn_label: str,
                     file_filter: Optional[str] = None,
                     extra_instruction: Optional[str] = None):
        def _work():
            try:
                sys.path.insert(0, str(Path(__file__).parent))
                from retriever.pipeline import run_pipeline
                from assistant.prompts import build_prompt
                from assistant.llm import stream_response

                result = run_pipeline(query, top_k=5)
                system, user_msg = build_prompt(
                    task, query, result.context, result.sources,
                    extra_instruction=extra_instruction)

                full = ""
                for token in stream_response(system, user_msg):
                    full += token
                    if len(full) % 60 == 0:
                        self._stream_queue.put(("generic_token", output_box, full))
                self._stream_queue.put(("generic_token", output_box, full))
                self._stream_queue.put(("generic_done", output_box, btn, btn_label))
            except Exception as e:
                self._stream_queue.put(("generic_token", output_box, f"❌ Error: {e}"))
                self._stream_queue.put(("generic_done", output_box, btn, btn_label))

        threading.Thread(target=_work, daemon=True).start()

    # ── Stream queue drain (runs on main thread via after()) ──────────────────
    def _drain_stream_queue(self):
        try:
            while True:
                item = self._stream_queue.get_nowait()
                kind = item[0]

                if kind == "chat_token":
                    _, box, text = item
                    box.configure(state="normal")
                    box.delete("1.0", "end")
                    box.insert("1.0", text)
                    self._resize_textbox(box, text)
                    box.configure(state="disabled")
                    self._chat_scroll._parent_canvas.yview_moveto(1.0)

                elif kind == "chat_done":
                    pass

                elif kind == "ask_trace":
                    self._set_output(self._ask_trace_text, item[1])

                elif kind == "ask_sources":
                    self._ask_sources_lbl.configure(text=item[1])

                elif kind == "ask_token":
                    # Drain any additional ask_token items that queued up
                    # between ticks — only render the most recent text.
                    latest = item[1]
                    try:
                        while True:
                            peeked = self._stream_queue.get_nowait()
                            if peeked[0] == "ask_token":
                                latest = peeked[1]
                            else:
                                # Put non-ask_token back at front (re-queue)
                                self._stream_queue.put(peeked)
                                break
                    except queue.Empty:
                        pass
                    self._set_output(self._ask_output, latest or "⏳ Waiting for model…")

                elif kind == "ask_done":
                    self._ask_btn.configure(state="normal", text="Ask")
                    self._poll_status()

                elif kind == "index_log":
                    self._append_log(self._index_log, item[1])

                elif kind == "index_progress":
                    _, pct, label, indexed_so_far = item
                    self._index_pbar.set(pct)
                    self._index_plabel.configure(text=label)
                    # Live-update the status bar chunk count without a DB round-trip
                    if indexed_so_far:
                        self._lbl_chunks.configure(text=f"📦 {indexed_so_far:,} chunks (indexing…)")

                elif kind == "index_done":
                    self._index_pbar.set(1.0)
                    self._index_plabel.configure(text=item[1])
                    self._index_btn.configure(state="normal", text="▶  Start Indexing")
                    self._index_stop_btn.configure(state="disabled")
                    self._indexing = False
                    # Show final chunk count from the message itself, then do a full poll
                    total_str = item[1]  # e.g. "Complete — 90,770 total chunks indexed"
                    # Extract the number using split — avoids regex escape issues
                    # Format: "Complete — 90,770 total chunks indexed"
                    _num = ""
                    for _w in total_str.split():
                        if _w.replace(",", "").isdigit():
                            _num = _w; break
                    if _num:
                        self._lbl_chunks.configure(text=f"📦 {_num} chunks")
                    self._poll_status()

                elif kind == "generic_token":
                    _, box, text = item
                    self._set_output(box, text)

                elif kind == "generic_done":
                    _, box, btn, label = item
                    btn.configure(state="normal", text=label)

                elif kind == "status":
                    _, ollama_ok, model_ok, model, chunks = item
                    icon = "✅" if model_ok else ("⚠️" if ollama_ok else "❌")
                    self._lbl_ollama.configure(
                        text=f"{icon} {model if model_ok else 'Model not pulled' if ollama_ok else 'Ollama offline'}")
                    self._lbl_chunks.configure(
                        text=f"📦 {chunks:,} chunks" if chunks else "")

        except queue.Empty:
            pass
        self.after(50, self._drain_stream_queue)

    def _set_output(self, box: ctk.CTkTextbox, text: str):
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", text)
        box.configure(state="disabled")

    def _append_log(self, box: ctk.CTkTextbox, text: str):
        box.configure(state="normal")
        box.insert("end", text)
        box.see("end")
        box.configure(state="disabled")

    def _browse_dir(self, entry: ctk.CTkEntry):
        d = filedialog.askdirectory()
        if d:
            entry.delete(0, "end")
            entry.insert(0, d)

    def _browse_file(self, entry: ctk.CTkEntry):
        f = filedialog.askopenfilename()
        if f:
            entry.delete(0, "end")
            entry.insert(0, f)

    def _poll_status(self):
        def _check():
            try:
                import requests as req
                sys.path.insert(0, str(Path(__file__).parent))
                import config.settings as s
                r = req.get(f"{s.OLLAMA_BASE_URL}/api/tags", timeout=3)
                ollama_ok = True
                models = [m["name"] for m in r.json().get("models", [])]
                model_ok = any(s.MODEL in m for m in models)
            except Exception:
                ollama_ok = model_ok = False
                s = type("s", (), {"MODEL": "?"})()
            try:
                from indexer.embedder import get_collection_stats
                stats = get_collection_stats()
                chunks = stats["total_chunks"]
            except Exception:
                chunks = 0
            self._stream_queue.put(("status", ollama_ok, model_ok,
                                    getattr(s, "MODEL", "?"), chunks))

        threading.Thread(target=_check, daemon=True).start()
        # Poll more frequently while indexing so the status bar stays fresh
        interval = 3_000 if self._indexing else 10_000
        self.after(interval, self._poll_status)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = App()
    app.mainloop()
