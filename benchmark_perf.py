from __future__ import annotations

import shutil
import time
from pathlib import Path

from indexer.scanner import scan_repo
from indexer.strategy import analyze_repo, format_analysis, get_profile


ROOT = Path("_perf_synthetic_repo").resolve()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_repo() -> int:
    shutil.rmtree(ROOT, ignore_errors=True)
    start = time.perf_counter()

    write(ROOT / "backend" / "pom.xml", "<project></project>\n")
    write(ROOT / "backend" / "src" / "main" / "resources" / "application.yml", "server:\n  port: 8080\n")
    write(ROOT / "framework" / "build.gradle", "plugins { id 'java-library' }\n")
    write(ROOT / "ui" / "package.json", '{"scripts":{}}\n')
    write(ROOT / "ui" / "angular.json", "{}\n")

    for i in range(6000):
        write(
            ROOT / "backend" / "src" / "main" / "java" / "com" / "example" / "service" / f"Service{i}.java",
            f"package com.example;\n@Service\npublic class Service{i} {{\n  public String run() {{ return \"ok\"; }}\n}}\n",
        )
    for i in range(3000):
        write(
            ROOT / "framework" / "src" / "main" / "java" / "com" / "example" / "core" / f"Core{i}.java",
            f"package com.example.core;\npublic class Core{i} {{\n  public void run() {{}}\n}}\n",
        )
    for i in range(3000):
        write(ROOT / "ui" / "src" / "app" / f"component{i}.ts", f"export const value{i} = {i};\n")
    for i in range(6000):
        write(ROOT / "backend" / "target" / "classes" / f"Generated{i}.class", "compiled-or-generated\n")
    for i in range(6000):
        write(ROOT / "ui" / "node_modules" / "pkg" / f"lib{i}.js", "vendor\n")

    elapsed = time.perf_counter() - start
    total_files = sum(1 for _ in ROOT.rglob("*") if _.is_file())
    print(f"created_files={total_files} seconds={elapsed:.2f}")
    return total_files


def time_call(label: str, fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"{label}_seconds={elapsed:.3f}")
    return result, elapsed


def main() -> None:
    build_repo()

    analysis, _ = time_call("analyze", lambda: analyze_repo(ROOT))
    print(format_analysis(analysis))

    for area in analysis.areas:
        profile = get_profile(area.profile)
        files, elapsed = time_call(
            f"scan_{area.profile}_{area.path.name}",
            lambda area=area, profile=profile: scan_repo(
                area.path,
                incremental=False,
                force_reindex=True,
                include_extensions=profile.include_extensions,
                include_filenames=profile.include_filenames,
                extra_skip_dirs=profile.extra_skip_dirs,
                max_file_size_kb=profile.max_file_size_kb,
                update_cache=False,
            )[0],
        )
        rate = len(files) / elapsed if elapsed else 0
        print(f"scan_{area.profile}_{area.path.name}_files={len(files)}")
        print(f"scan_{area.profile}_{area.path.name}_files_per_sec={rate:.0f}")

    shutil.rmtree(ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
