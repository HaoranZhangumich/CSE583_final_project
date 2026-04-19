from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
COARSENING_FACTORS = [1, 2, 4, 8, 16, 32]
DEVICE_LABEL_TO_ID = {"CPU": 0, "GPU": 1}
DEVICE_ID_TO_LABEL = {0: "CPU", 1: "GPU"}
CF_TO_CLASS = {cf: i for i, cf in enumerate(COARSENING_FACTORS)}
CLASS_TO_CF = {i: cf for i, cf in enumerate(COARSENING_FACTORS)}


@dataclasses.dataclass
class TrainConfig:
    input_mode: str = "ir"          # "ir" or "source"
    model_type: str = "transformer" # "transformer" or "lstm"

    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4

    d_model: int = 64
    nhead: int = 8
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1

    max_program_len: int = 1024
    max_stmt_len: int = 64

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Dict) -> None:
    write_text(path, json.dumps(obj, indent=2))


def unique_in_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def load_device_mapping_df(repo_root: Path, platform: str) -> pd.DataFrame:
    csv_path = repo_root / "data" / "case-study-a" / f"cgo17-{platform}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing dataset file: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"benchmark", "oracle", "runtime_cpu", "runtime_gpu", "src"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {sorted(missing)}")
    return df


def load_thread_coarsening_data(repo_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    runtimes_path = repo_root / "data" / "case-study-b" / "pact-2014-runtimes.csv"
    oracles_path = repo_root / "data" / "case-study-b" / "pact-2014-oracles.csv"
    if not runtimes_path.exists() or not oracles_path.exists():
        raise FileNotFoundError(
            f"Missing thread-coarsening files: {runtimes_path} / {oracles_path}"
        )
    runtimes = pd.read_csv(runtimes_path)
    oracles = pd.read_csv(oracles_path)
    if "src" not in runtimes.columns or "kernel" not in runtimes.columns:
        raise ValueError("Thread coarsening runtimes CSV must contain 'kernel' and 'src'.")
    return runtimes, oracles


def collect_all_unique_sources(repo_root: Path) -> List[str]:
    srcs: List[str] = []
    for platform in ["amd", "nvidia"]:
        df = load_device_mapping_df(repo_root, platform)
        srcs.extend(df["src"].astype(str).tolist())
    runtimes, _ = load_thread_coarsening_data(repo_root)
    srcs.extend(runtimes["src"].astype(str).tolist())
    return unique_in_order(srcs)
