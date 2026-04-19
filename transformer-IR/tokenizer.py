from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.normalizers import Sequence as NormalizerSequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

from config import SPECIAL_TOKENS, collect_all_unique_sources, ensure_dir, read_text


def iter_text_statements(repo_root: Path, cache_dir: Path, input_mode: str) -> Iterable[str]:
    if input_mode == "ir":
        for path in sorted(cache_dir.glob("*.ll")):
            ir_text = read_text(path)
            for line in ir_text.splitlines():
                stmt = line.strip()
                if stmt:
                    yield stmt
    elif input_mode == "source":
        for src in collect_all_unique_sources(repo_root):
            for line in str(src).splitlines():
                stmt = line.strip()
                if stmt:
                    yield stmt
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")


def train_wordpiece_tokenizer(
    repo_root: Path,
    cache_dir: Path,
    tokenizer_path: Path,
    input_mode: str,
    vocab_size: int = 8000,
    min_frequency: int = 2,
) -> Path:
    ensure_dir(tokenizer_path.parent)
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = NormalizerSequence([NFD(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(
        iter_text_statements(repo_root, cache_dir, input_mode),
        trainer=trainer,
    )

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )
    tokenizer.save(str(tokenizer_path))
    return tokenizer_path


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    return Tokenizer.from_file(str(tokenizer_path))


def encode_program_text(
    tokenizer: Tokenizer,
    text: str,
    max_stmt_len: int = 64,
    max_program_len: int = 1024,
) -> List[int]:
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    pad_id = tokenizer.token_to_id("[PAD]")
    if cls_id is None or sep_id is None or pad_id is None:
        raise ValueError("Tokenizer is missing required special tokens.")

    tokens: List[int] = [cls_id]
    for stmt in str(text).splitlines():
        stmt = stmt.strip()
        if not stmt:
            continue
        enc = tokenizer.encode(stmt)
        stmt_ids = enc.ids
        if len(stmt_ids) >= 2 and stmt_ids[0] == cls_id and stmt_ids[-1] == sep_id:
            stmt_ids = stmt_ids[1:-1]
        stmt_ids = stmt_ids[: max(0, max_stmt_len - 1)]
        tokens.extend(stmt_ids)
        tokens.append(sep_id)
        if len(tokens) >= max_program_len:
            tokens = tokens[:max_program_len]
            break

    if len(tokens) < max_program_len:
        tokens.extend([pad_id] * (max_program_len - len(tokens)))
    else:
        tokens = tokens[:max_program_len]
    return tokens
