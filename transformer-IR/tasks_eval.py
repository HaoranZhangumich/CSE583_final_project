from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tqdm import tqdm

from config import (
    CF_TO_CLASS,
    CLASS_TO_CF,
    COARSENING_FACTORS,
    DEVICE_ID_TO_LABEL,
    DEVICE_LABEL_TO_ID,
    TrainConfig,
    ensure_dir,
    load_device_mapping_df,
    load_thread_coarsening_data,
)
from ir_pipeline import load_ir_for_source, has_ir_for_source
from model import predict, train_one_model
from tokenizer import encode_program_text, load_tokenizer


def _has_input_for_source(src: str, cache_dir: Path, input_mode: str) -> bool:
    if input_mode == "source":
        return True
    if input_mode == "ir":
        return has_ir_for_source(src, cache_dir)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def _load_program_text(src: str, cache_dir: Path, input_mode: str) -> str:
    if input_mode == "source":
        return str(src)
    if input_mode == "ir":
        return load_ir_for_source(src, cache_dir)
    raise ValueError(f"Unsupported input_mode: {input_mode}")


def prepare_device_mapping_inputs(
    repo_root: Path,
    cache_dir: Path,
    tokenizer,
    platform: str,
    cfg: TrainConfig,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if isinstance(tokenizer, (str, Path)):
        tokenizer = load_tokenizer(Path(tokenizer))

    df = load_device_mapping_df(repo_root, platform)

    kept_rows: List[int] = []
    sequences = []
    labels = []
    skipped = 0

    for i, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Encoding {cfg.input_mode} ({platform})",
    ):
        src = str(row["src"])

        if not _has_input_for_source(src, cache_dir, cfg.input_mode):
            skipped += 1
            continue

        program_text = _load_program_text(src, cache_dir, cfg.input_mode)
        sequences.append(
            encode_program_text(tokenizer, program_text, cfg.max_stmt_len, cfg.max_program_len)
        )
        labels.append(DEVICE_LABEL_TO_ID[str(row["oracle"])])
        kept_rows.append(i)

    filtered_df = df.iloc[kept_rows].reset_index(drop=True)
    labels = np.asarray(labels, dtype=np.int64)
    sequences = np.asarray(sequences, dtype=np.int64)

    print(
        f"[device-mapping:{platform}:{cfg.input_mode}] kept {len(filtered_df)} "
        f"samples, skipped {skipped} unavailable samples."
    )
    print(f"[device-mapping:{platform}] label counts after filtering: {np.bincount(labels).tolist()}")

    if len(filtered_df) == 0:
        raise RuntimeError(f"No valid samples left for platform={platform}.")

    return filtered_df, sequences, labels


def clamp_cf_to_available(pred_cf: int, available_cfs: Sequence[int]) -> int:
    available_sorted = sorted(set(int(x) for x in available_cfs))
    candidates = [cf for cf in available_sorted if cf <= pred_cf]
    return max(candidates) if candidates else available_sorted[0]


def prepare_thread_coarsening_inputs(
    repo_root: Path,
    cache_dir: Path,
    tokenizer,
    cfg: TrainConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], np.ndarray]:
    if isinstance(tokenizer, (str, Path)):
        tokenizer = load_tokenizer(Path(tokenizer))

    runtimes, oracles = load_thread_coarsening_data(repo_root)
    kernels = sorted(set(runtimes["kernel"].astype(str)))

    kernel_to_src: Dict[str, str] = {}
    for kernel in kernels:
        rows = runtimes[runtimes["kernel"] == kernel]
        kernel_to_src[kernel] = str(rows.iloc[0]["src"])

    kept_kernels: List[str] = []
    sequences = []
    skipped = 0

    for kernel in tqdm(kernels, desc=f"Encoding {cfg.input_mode} (thread coarsening)"):
        src = kernel_to_src[kernel]

        if not _has_input_for_source(src, cache_dir, cfg.input_mode):
            skipped += 1
            continue

        program_text = _load_program_text(src, cache_dir, cfg.input_mode)
        sequences.append(
            encode_program_text(tokenizer, program_text, cfg.max_stmt_len, cfg.max_program_len)
        )
        kept_kernels.append(kernel)

    filtered_runtimes = runtimes[runtimes["kernel"].astype(str).isin(kept_kernels)].copy()
    filtered_oracles = oracles[oracles["kernel"].astype(str).isin(kept_kernels)].copy()
    filtered_oracles = filtered_oracles.set_index("kernel").loc[kept_kernels].reset_index()

    print(
        f"[thread-coarsening:{cfg.input_mode}] kept {len(kept_kernels)} kernels, "
        f"skipped {skipped} unavailable samples."
    )

    if len(kept_kernels) == 0:
        raise RuntimeError("No valid kernels left for thread coarsening.")

    return filtered_runtimes, filtered_oracles, kept_kernels, np.asarray(sequences, dtype=np.int64)


def _make_inner_train_val_split(
    train_idx: np.ndarray,
    labels: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    train_labels = labels[train_idx]
    n_train = len(train_idx)

    # too few samples: just reuse train as val
    if n_train < 4:
        return train_idx, train_idx

    class_counts = np.bincount(train_labels)
    present_classes = int(np.count_nonzero(class_counts))
    min_class_count = int(class_counts[class_counts > 0].min()) if present_classes > 0 else 0

    # Try stratified split only when it is actually feasible:
    # 1) every present class has at least 2 samples
    # 2) val set size >= number of present classes
    # 3) train set size after split >= number of present classes
    if present_classes >= 2 and min_class_count >= 2:
        test_size = max(1, int(round(0.15 * n_train)))
        test_size = max(test_size, present_classes)

        max_allowed_test_size = n_train - present_classes
        if max_allowed_test_size >= present_classes:
            test_size = min(test_size, max_allowed_test_size)

            if test_size >= present_classes and (n_train - test_size) >= present_classes:
                inner_train_idx, val_idx = train_test_split(
                    train_idx,
                    test_size=test_size,
                    random_state=seed,
                    stratify=train_labels,
                )
                return np.asarray(inner_train_idx), np.asarray(val_idx)

    # fallback: non-stratified random split
    shuffled = train_idx.copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)

    split = max(1, int(round(0.85 * n_train)))
    split = min(split, n_train - 1)

    return shuffled[:split], shuffled[split:]


def run_device_mapping_cv(
    repo_root: Path,
    cache_dir: Path,
    tokenizer_path: Path,
    out_dir: Path,
    seed: int,
    cfg: TrainConfig,
) -> pd.DataFrame:
    ensure_dir(out_dir)

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")

    rows = []

    for platform in ["amd", "nvidia"]:
        df, sequences, labels = prepare_device_mapping_inputs(
            repo_root, cache_dir, tokenizer, platform, cfg
        )

        class_counts = np.bincount(labels)
        min_class_count = int(class_counts.min())
        n_splits = min(10, min_class_count)

        if n_splits < 2:
            raise RuntimeError(
                f"Not enough valid samples for stratified CV on {platform}. "
                f"class_counts={class_counts.tolist()}"
            )

        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        print(
            f"[device-mapping:{platform}:{cfg.input_mode}:{cfg.model_type}] "
            f"using {n_splits}-fold CV after filtering."
        )

        for fold, (train_idx, test_idx) in enumerate(kf.split(sequences, labels)):
            inner_train_idx, val_idx = _make_inner_train_val_split(
                train_idx=train_idx,
                labels=labels,
                seed=seed + fold,
            )

            model = train_one_model(
                train_sequences=sequences[inner_train_idx],
                train_labels=labels[inner_train_idx],
                val_sequences=sequences[val_idx],
                val_labels=labels[val_idx],
                vocab_size=vocab_size,
                num_classes=2,
                pad_id=pad_id,
                cfg=cfg,
                use_class_weights=True,
            )
            preds = predict(model, sequences[test_idx], cfg)

            print(
                f"[device-mapping:{platform}] fold={fold} "
                f"train_counts={np.bincount(labels[inner_train_idx], minlength=2).tolist()} "
                f"val_counts={np.bincount(labels[val_idx], minlength=2).tolist()} "
                f"test_counts={np.bincount(labels[test_idx], minlength=2).tolist()} "
                f"pred_counts={np.bincount(preds, minlength=2).tolist()}"
            )

            for row_idx, pred in zip(test_idx, preds):
                runtime_cpu = float(df.iloc[row_idx]["runtime_cpu"])
                runtime_gpu = float(df.iloc[row_idx]["runtime_gpu"])
                baseline_runtime = runtime_cpu if platform == "amd" else runtime_gpu
                pred_runtime = runtime_cpu if pred == 0 else runtime_gpu
                rows.append(
                    {
                        "task": "device_mapping",
                        "platform": platform,
                        "fold": fold,
                        "benchmark": df.iloc[row_idx]["benchmark"],
                        "oracle_label": int(labels[row_idx]),
                        "pred_label": int(pred),
                        "oracle_name": DEVICE_ID_TO_LABEL[int(labels[row_idx])],
                        "pred_name": DEVICE_ID_TO_LABEL[int(pred)],
                        "correct": int(pred == labels[row_idx]),
                        "speedup_vs_default": baseline_runtime / pred_runtime if pred_runtime > 0 else np.nan,
                        "input_mode": cfg.input_mode,
                        "model_type": cfg.model_type,
                    }
                )

    result_df = pd.DataFrame(rows)
    result_path = out_dir / f"device_mapping_results_{cfg.input_mode}_{cfg.model_type}.csv"
    result_df.to_csv(result_path, index=False)
    return result_df


def run_thread_coarsening_cv(
    repo_root: Path,
    cache_dir: Path,
    tokenizer_path: Path,
    out_dir: Path,
    cfg: TrainConfig,
) -> pd.DataFrame:
    ensure_dir(out_dir)

    tokenizer = load_tokenizer(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    pad_id = tokenizer.token_to_id("[PAD]")

    runtimes, oracles, kernels, sequences = prepare_thread_coarsening_inputs(
        repo_root, cache_dir, tokenizer, cfg
    )

    rows = []
    for platform in ["Fermi", "Kepler", "Cypress", "Tahiti"]:
        labels = np.asarray(
            [CF_TO_CLASS[int(x)] for x in oracles[f"cf_{platform}"]],
            dtype=np.int64,
        )
        kf = KFold(n_splits=len(kernels), shuffle=False)

        for fold, (train_idx, test_idx) in enumerate(kf.split(sequences)):
            inner_train_idx, val_idx = _make_inner_train_val_split(
                train_idx=train_idx,
                labels=labels,
                seed=1000 + fold,
            )

            model = train_one_model(
                train_sequences=sequences[inner_train_idx],
                train_labels=labels[inner_train_idx],
                val_sequences=sequences[val_idx],
                val_labels=labels[val_idx],
                vocab_size=vocab_size,
                num_classes=len(COARSENING_FACTORS),
                pad_id=pad_id,
                cfg=cfg,
                use_class_weights=False,
            )
            pred_class = int(predict(model, sequences[test_idx], cfg)[0])
            pred_cf = CLASS_TO_CF[pred_class]
            kernel = kernels[int(test_idx[0])]
            kernel_rows = runtimes[runtimes["kernel"] == kernel]
            available_cfs = kernel_rows["cf"].astype(int).tolist()
            pred_cf = clamp_cf_to_available(pred_cf, available_cfs)
            oracle_cf = int(oracles.iloc[int(test_idx[0])][f"cf_{platform}"])

            no_cf_runtime = float(
                kernel_rows[kernel_rows["cf"].astype(int) == 1][f"runtime_{platform}"].iloc[0]
            )
            pred_runtime = float(
                kernel_rows[kernel_rows["cf"].astype(int) == pred_cf][f"runtime_{platform}"].iloc[0]
            )
            oracle_runtime = float(oracles.iloc[int(test_idx[0])][f"runtime_{platform}"])

            rows.append(
                {
                    "task": "thread_coarsening",
                    "platform": platform,
                    "fold": fold,
                    "kernel": kernel,
                    "oracle_cf": oracle_cf,
                    "pred_cf": pred_cf,
                    "correct": int(pred_cf == oracle_cf),
                    "speedup_vs_no_cf": no_cf_runtime / pred_runtime if pred_runtime > 0 else np.nan,
                    "oracle_fraction": oracle_runtime / pred_runtime if pred_runtime > 0 else np.nan,
                    "input_mode": cfg.input_mode,
                    "model_type": cfg.model_type,
                }
            )

    result_df = pd.DataFrame(rows)
    result_path = out_dir / f"thread_coarsening_results_{cfg.input_mode}_{cfg.model_type}.csv"
    result_df.to_csv(result_path, index=False)
    return result_df


def summarize_results(out_dir: Path, cfg: TrainConfig) -> None:
    dm_path = out_dir / f"device_mapping_results_{cfg.input_mode}_{cfg.model_type}.csv"
    tc_path = out_dir / f"thread_coarsening_results_{cfg.input_mode}_{cfg.model_type}.csv"

    if dm_path.exists():
        dm = pd.read_csv(dm_path)
        print("\n=== Device Mapping Summary ===")
        print(dm.groupby("platform")[["correct", "speedup_vs_default"]].mean())

    if tc_path.exists():
        tc = pd.read_csv(tc_path)
        print("\n=== Thread Coarsening Summary ===")
        print(tc.groupby("platform")[["correct", "speedup_vs_no_cf", "oracle_fraction"]].mean())
