from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from config import collect_all_unique_sources, ensure_dir, read_text, sha1_text, write_json, write_text


def default_clang_commands() -> List[List[str]]:
    return [
        [
            "clang",
            "-x",
            "cl",
            "-cl-std=CL1.2",
            "-target",
            "spir64",
            "-Xclang",
            "-finclude-default-header",
            "-emit-llvm",
            "-S",
            "{src}",
            "-o",
            "{out}",
        ],
        [
            "clang",
            "-cc1",
            "-x",
            "cl",
            "-cl-std=CL1.2",
            "-triple",
            "spir64-unknown-unknown",
            "-finclude-default-header",
            "-emit-llvm",
            "-O0",
            "-disable-llvm-passes",
            "-o",
            "{out}",
            "{src}",
        ],
    ]


def extract_function_bodies(ir_text: str) -> str:
    kept: List[str] = []
    inside = False
    brace_depth = 0

    for raw_line in ir_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("define "):
            inside = True
            brace_depth = stripped.count("{") - stripped.count("}")
            kept.append(line)
            continue

        if inside:
            kept.append(line)
            brace_depth += stripped.count("{") - stripped.count("}")
            if brace_depth <= 0:
                inside = False

    return "\n".join(kept).strip()


def normalize_llvm_ir(ir_text: str) -> str:
    lines: List[str] = []
    for line in ir_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("!"):
            continue
        if stripped.startswith("; ModuleID ="):
            continue
        if stripped.startswith("source_filename ="):
            continue
        if stripped.startswith("attributes #"):
            continue
        if stripped.startswith("declare "):
            continue
        stripped = re.sub(r",?\s*!dbg\s*!\d+", "", stripped)
        stripped = re.sub(r",?\s*!tbaa\s*!\d+", "", stripped)
        stripped = re.sub(r",?\s*!range\s*!\d+", "", stripped)
        stripped = re.sub(r",?\s*!llvm\.loop\s*!\d+", "", stripped)
        stripped = re.sub(r"#[0-9]+", "", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def compile_opencl_to_llvm_ir(src_code: str, clang_bin: str = "clang") -> str:
    with tempfile.TemporaryDirectory(prefix="deeptune_ir_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        src_path = tmpdir_path / "kernel.cl"
        out_path = tmpdir_path / "kernel.ll"
        write_text(src_path, src_code)

        errors: List[str] = []
        for command_template in default_clang_commands():
            cmd = [clang_bin if x == "clang" else x for x in command_template]
            cmd = [arg.format(src=str(src_path), out=str(out_path)) for arg in cmd]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "clang was not found. Please install clang and make sure it is on PATH."
                ) from exc

            if proc.returncode == 0 and out_path.exists():
                raw_ir = read_text(out_path)
                # to keep more information for IR
                body_ir = extract_function_bodies(raw_ir)
                ir_to_use = body_ir if body_ir else raw_ir
                normalized = normalize_llvm_ir(raw_ir)
                if normalized:
                    return normalized
                return normalize_llvm_ir(raw_ir)

            errors.append(
                f"Command failed: {' '.join(cmd)}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        raise RuntimeError("Failed to compile OpenCL source to LLVM IR.\n\n" + "\n\n".join(errors))


def build_ir_cache(repo_root: Path, cache_dir: Path, clang_bin: str = "clang") -> Path:
    ensure_dir(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    manifest: Dict[str, Dict[str, str]] = {}
    if manifest_path.exists():
        manifest = json.loads(read_text(manifest_path))

    sources = collect_all_unique_sources(repo_root)
    for src_code in tqdm(sources, desc="Compiling LLVM IR"):
        src_hash = sha1_text(src_code)
        ir_path = cache_dir / f"{src_hash}.ll"
        if ir_path.exists():
            manifest[src_hash] = {"ir_path": str(ir_path), "status": "ok"}
            continue
        try:
            ir_text = compile_opencl_to_llvm_ir(src_code, clang_bin=clang_bin)
            write_text(ir_path, ir_text)
            manifest[src_hash] = {"ir_path": str(ir_path), "status": "ok"}
        except Exception as exc:  # noqa: BLE001
            manifest[src_hash] = {"ir_path": "", "status": "error", "error": str(exc)}

    write_json(manifest_path, manifest)

    num_ok = sum(1 for v in manifest.values() if v.get("status") == "ok")
    num_err = sum(1 for v in manifest.values() if v.get("status") == "error")
    print(f"IR cache build finished: ok={num_ok}, error={num_err}")
    return manifest_path


def has_ir_for_source(src_code: str, cache_dir: Path) -> bool:
    src_hash = sha1_text(src_code)
    ir_path = cache_dir / f"{src_hash}.ll"
    return ir_path.exists()


def load_ir_for_source(src_code: str, cache_dir: Path) -> str:
    src_hash = sha1_text(src_code)
    ir_path = cache_dir / f"{src_hash}.ll"
    if not ir_path.exists():
        raise FileNotFoundError(
            f"Missing cached IR for source hash {src_hash}. Run build_ir_cache() first."
        )
    return read_text(ir_path)