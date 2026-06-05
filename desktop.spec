# desktop.spec
# PyInstaller build spec for Coding Assistant Windows desktop app.
#
# Build command (run from project root with venv activated):
#   pyinstaller desktop.spec
#
# Output:  dist/CodingAssistant.exe  (~80-120 MB single file)

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules
import sys, os

block_cipher = None

# Collect all submodules for chromadb, tree_sitter, etc.
datas = []
binaries = []
hiddenimports = []

for pkg in ['chromadb', 'hnswlib', 'tokenizers', 'fastapi', 'uvicorn',
            'starlette', 'anyio', 'sniffio', 'tree_sitter', 'tree_sitter_languages',
            'webview']:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception:
        pass

# Include the UI folder
datas += [('ui', 'ui')]

a = Analysis(
    ['desktop.py'],
    pathex=[os.path.abspath('.')],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + [
        'assistant', 'assistant.llm', 'assistant.prompts',
        'config', 'config.settings',
        'indexer', 'indexer.scanner', 'indexer.chunker',
        'indexer.embedder', 'indexer.strategy',
        'retriever', 'retriever.hybrid_search',
        'retriever.context_builder', 'retriever.pipeline',
        'cli', 'cli.main',
        'uvicorn.logging', 'uvicorn.loops', 'uvicorn.loops.auto',
        'uvicorn.protocols', 'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto', 'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto', 'uvicorn.lifespan',
        'uvicorn.lifespan.on',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'notebook', 'ipython', 'PIL', 'cv2', 'torch'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
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
    name='CodingAssistant',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # No console window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='ui/icon.ico',     # optional icon
    version='version_info.txt',  # optional Windows version resource
)
