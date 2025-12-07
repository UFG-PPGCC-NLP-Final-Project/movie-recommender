# redial_dataset_eval.py
#
# Dataset e Collate exclusivos para avaliação,
# para não interferir no código usado no treino.

import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer


class RedialEvalDataset(Dataset):
    """
    Lê processed/test.jsonl, processed/valid.jsonl ou train.jsonl para avaliação.
    """

    def __init__(self, jsonl_path: Path):
        self.examples: List[Dict[str, Any]] = []

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                self.examples.append(obj)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def redial_eval_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: BertTokenizer,
    max_len: int,
    num_movies: int,
):
    """
    Collate alternativo, projetado só para avaliação.
    Não interfere no collate de treino.
    """

    # ------------------- BERT -------------------
    texts = [ex["bert_text"] for ex in batch]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    token_type_ids = enc["token_type_ids"]

    # ------------------- RNN --------------------
    rnn_movies_list = [ex["rnn_movies"] for ex in batch]
    batch_size = len(rnn_movies_list)
    max_seq_len = max(len(seq) for seq in rnn_movies_list) if rnn_movies_list else 1

    rnn_input = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    rnn_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    for i, seq in enumerate(rnn_movies_list):
        shifted = [m + 1 for m in seq]  # +1 para reservar 0 como PAD
        if len(shifted) > 0:
            rnn_input[i, :len(shifted)] = torch.tensor(shifted, dtype=torch.long)
            rnn_mask[i, :len(shifted)] = True

    # ------------------- LABELS -----------------
    labels_list = [ex["labels"] for ex in batch]
    labels = torch.zeros((batch_size, num_movies), dtype=torch.float32)
    for i, lbls in enumerate(labels_list):
        for m in lbls:
            if 0 <= m < num_movies:
                labels[i, m] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "rnn_input": rnn_input,
        "rnn_mask": rnn_mask,
        "labels": labels,
        "rnn_movies_list": rnn_movies_list,  # ← usado nas métricas
        "rnn_vocab_size": num_movies + 1,
    }

    
