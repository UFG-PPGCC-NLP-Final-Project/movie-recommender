import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class RedialDataset(Dataset):
    """
    Lê o processed/*.jsonl gerado antes.

    Cada linha do jsonl tem:
      - bert_text: string
      - rnn_movies: [idx_movie, ...]
      - labels: [idx_movie_label, ...]
    """

    def __init__(self, jsonl_path: Path):
        self.jsonl_path = jsonl_path
        self.examples: List[Dict[str, Any]] = []

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.examples.append(obj)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        return {
            "bert_text": ex["bert_text"],
            "rnn_movies": ex["rnn_movies"],
            "labels": ex["labels"],
        }


def redial_collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: BertTokenizer,
    max_len: int,
    num_movies: int,
):
    """
    Collate que:

    - tokeniza bert_text
    - cria sequência de filmes para RNN (com padding)
    - cria vetor multi-hot de labels
    """
    # --------- TEXTOS PARA BERT ----------
    texts = [item["bert_text"] for item in batch]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]          # (B, L)
    attention_mask = enc["attention_mask"]
    token_type_ids = enc["token_type_ids"]

    # --------- SEQUÊNCIA DE FILMES (RNN) ----------
    movie_seqs = [item["rnn_movies"] for item in batch]
    batch_size = len(movie_seqs)
    max_seq_len = max(len(seq) for seq in movie_seqs)

    # truque: vamos usar 0 como PAD, então deslocamos todos índices +1
    rnn_input = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    rnn_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.bool)

    for i, seq in enumerate(movie_seqs):
        if not seq:
            continue
        shifted = [x + 1 for x in seq]  # +1 para reservar 0 como PAD
        length = len(shifted)
        rnn_input[i, :length] = torch.tensor(shifted, dtype=torch.long)
        rnn_mask[i, :length] = True

    # Vocabulário para a RNN = num_movies + 1 (por causa do shift)
    rnn_vocab_size = num_movies + 1

    # --------- LABELS MULTI-HOT ----------
    labels_list = [item["labels"] for item in batch]
    labels = torch.zeros((batch_size, num_movies), dtype=torch.float32)

    for i, lbls in enumerate(labels_list):
        for idx in lbls:
            if 0 <= idx < num_movies:
                labels[i, idx] = 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "rnn_input": rnn_input,
        "rnn_mask": rnn_mask,
        "labels": labels,
        "rnn_vocab_size": rnn_vocab_size,  # útil na definição do embedding da RNN
    }
