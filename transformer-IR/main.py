from __future__ import annotations

import argparse
from pathlib import Path

from config import TrainConfig, set_seed
from ir_pipeline import build_ir_cache
from tasks_eval import run_device_mapping_cv, run_thread_coarsening_cv, summarize_results
from tokenizer import train_wordpiece_tokenizer


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run IR/source - Transformer/LSTM experiments on paper-end2end-dl datasets."
    )
    parser.add_argument("--repo-root", type=Path, required=True, help="Path to the paper-end2end-dl repo.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./out"),
        help="Directory for IR cache, tokenizer, and results.",
    )
    parser.add_argument("--clang-bin", type=str, default="clang", help="Clang executable name or path.")
    parser.add_argument("--seed", type=int, default=204)

    parser.add_argument("--input-mode", choices=["ir", "source"], default="ir")
    parser.add_argument("--model-type", choices=["transformer", "lstm"], default="transformer")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--max-program-len", type=int, default=1024)
    parser.add_argument("--max-stmt-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--min-frequency", type=int, default=2)

    parser.add_argument(
        "--task",
        choices=["build-ir", "train-tokenizer", "device-mapping", "thread-coarsening", "all"],
        default="all",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()
    set_seed(args.seed)

    repo_root: Path = args.repo_root.resolve()
    out_dir: Path = args.out_dir.resolve()
    ir_cache_dir = out_dir / "ir_cache"
    tokenizer_path = out_dir / "tokenizer" / f"wordpiece_{args.input_mode}.json"
    results_dir = out_dir / "results"

    cfg = TrainConfig(
        input_mode=args.input_mode,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_program_len=args.max_program_len,
        max_stmt_len=args.max_stmt_len,
    )

    if args.task == "build-ir" and args.input_mode != "ir":
        raise ValueError("--task build-ir is only valid when --input-mode ir")

    if args.input_mode == "ir" and args.task in {"build-ir", "train-tokenizer", "device-mapping", "thread-coarsening", "all"}:
        manifest_path = build_ir_cache(repo_root, ir_cache_dir, clang_bin=args.clang_bin)
        print(f"IR cache manifest saved to: {manifest_path}")

    if args.task in {"train-tokenizer", "device-mapping", "thread-coarsening", "all"}:
        tokenizer_path = train_wordpiece_tokenizer(
            repo_root=repo_root,
            cache_dir=ir_cache_dir,
            tokenizer_path=tokenizer_path,
            input_mode=args.input_mode,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
        print(f"Tokenizer saved to: {tokenizer_path}")

    if args.task in {"device-mapping", "all"}:
        dm = run_device_mapping_cv(repo_root, ir_cache_dir, tokenizer_path, results_dir, args.seed, cfg)
        print(
            f"Saved device-mapping results to: "
            f"{results_dir / f'device_mapping_results_{cfg.input_mode}_{cfg.model_type}.csv'}"
        )
        print(dm.groupby("platform")[["correct", "speedup_vs_default"]].mean())

    if args.task in {"thread-coarsening", "all"}:
        tc = run_thread_coarsening_cv(repo_root, ir_cache_dir, tokenizer_path, results_dir, cfg)
        print(
            f"Saved thread-coarsening results to: "
            f"{results_dir / f'thread_coarsening_results_{cfg.input_mode}_{cfg.model_type}.csv'}"
        )
        print(tc.groupby("platform")[["correct", "speedup_vs_no_cf", "oracle_fraction"]].mean())

    summarize_results(results_dir, cfg)


if __name__ == "__main__":
    main()
