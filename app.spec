# app.spec
# PyInstaller build spec for CodingAssistant.exe
#
# Prerequisites (run in activated venv):
#   pip install pyinstaller customtkinter pillow
#
# Build:
#   pyinstaller app.spec
#
# Output: dist\CodingAssistant.exe  (~80-150 MB)
#
# Notes:
# - customtkinter bundles its own theme files; collect_data_files handles this
# - tkinter ships with Python on Windows — no extras needed
# - chromadb + hnswlib need their native .pyd/.dll collected

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_all

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# customtkinter needs its assets bundled
datas += collect_data_files("customtkinter")

# chromadb pulls in native libs
for pkg in ["chromadb", "hnswlib", "tokenizers"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

# tree-sitter language grammars
try:
    d, b, h = collect_all("tree_sitter_languages")
    datas += d; binaries += b; hiddenimports += h
except Exception:
    pass

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        "tkinter", "tkinter.filedialog", "tkinter.messagebox",
        "customtkinter",
        "assistant", "assistant.llm", "assistant.prompts",
        "config", "config.settings",
        "indexer", "indexer.scanner", "indexer.chunker",
        "indexer.embedder", "indexer.strategy",
        "retriever", "retriever.hybrid_search",
        "retriever.context_builder", "retriever.pipeline",
        "cli", "cli.main",
        "PIL", "PIL.Image",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=["matplotlib", "notebook", "ipython", "cv2", "torch",
              "fastapi", "uvicorn", "starlette"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="CodingAssistant",
    debug=False,
    strip=False,
    upx=True,
    console=False,        # No black console window on Windows
    windowed=True,
    icon="ui\\icon.ico",  # optional — remove line if no icon file
)
