from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import TrainConfig


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        max_len: int = 1024,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.positional = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids.eq(self.pad_id)
        x = self.embedding(input_ids)
        x = self.positional(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        non_pad = (~mask).unsqueeze(-1)
        summed = (x * non_pad).sum(dim=1)
        denom = non_pad.sum(dim=1).clamp(min=1)
        pooled = summed / denom
        pooled = self.norm(pooled)
        return self.head(pooled)


class SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[Sequence[int]], labels: Sequence[int]):
        self.sequences = np.asarray(sequences, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.sequences[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def train_one_model(
    train_sequences,
    train_labels,
    val_sequences,
    val_labels,
    vocab_size,
    num_classes,
    pad_id,
    cfg,
):
    model = TransformerClassifier(
        vocab_size=vocab_size,
        num_classes=num_classes,
        max_len=cfg.max_program_len,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        pad_id=pad_id,
    ).to(cfg.device)

    train_loader = DataLoader(
        SequenceDataset(train_sequences, train_labels),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(val_sequences, val_labels),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    best_state = None
    best_acc = -1.0

    epoch_bar = tqdm(range(cfg.epochs), desc="Training epochs", leave=True)

    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs} [train]",
            leave=False,
        )

        for step, batch in enumerate(train_bar, start=1):
            input_ids = batch["input_ids"].to(cfg.device)
            labels = batch["labels"].to(cfg.device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{running_loss / step:.4f}",
            )

        model.eval()
        correct = 0
        total = 0
        val_loss_sum = 0.0

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs} [val]",
            leave=False,
        )

        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch["input_ids"].to(cfg.device)
                labels = batch["labels"].to(cfg.device)

                logits = model(input_ids)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=-1)

                val_loss_sum += loss.item()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / max(total, 1)
        avg_train_loss = running_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss_sum / max(len(val_loader), 1)

        epoch_bar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}",
            val_loss=f"{avg_val_loss:.4f}",
            val_acc=f"{val_acc:.4f}",
            best=f"{max(best_acc, val_acc):.4f}",
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def predict(model: TransformerClassifier, sequences: Sequence[Sequence[int]], cfg: TrainConfig) -> np.ndarray:
    loader = DataLoader(SequenceDataset(sequences, [0] * len(sequences)), batch_size=cfg.batch_size, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(cfg.device)
            logits = model(input_ids)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    return np.asarray(preds, dtype=np.int64)
