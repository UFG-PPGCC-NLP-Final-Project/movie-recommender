# movielens_tags_dataset.py

import json
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset


class MovieLensTagsDataset(Dataset):
    """
    Lê movielens_tags.jsonl e mapeia movieId → índice interno de filme (mesmo
    índice do ReDial), e tag_ids → multi-hot em num_tags.

    Cada item:
      {
        "movie_idx": int (0..num_movies-1),
        "tag_ids": [int, ...]
      }
    """

    def __init__(
        self,
        tags_jsonl_path: Path,
        movie_id_to_index_path: Path,
    ):
        self.tags_jsonl_path = tags_jsonl_path
        self.movie_id_to_index_path = movie_id_to_index_path

        with movie_id_to_index_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        # chaves vêm como string
        self.movie_id_to_index: Dict[int, int] = {int(k): int(v) for k, v in raw.items()}

        self.examples: List[Dict[str, Any]] = []

        with tags_jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                movie_id = int(obj["movieId"])
                tag_ids = obj["tag_ids"]

                if movie_id not in self.movie_id_to_index:
                    continue

                movie_idx = self.movie_id_to_index[movie_id]
                if not tag_ids:
                    continue

                self.examples.append(
                    {
                        "movie_idx": movie_idx,
                        "tag_ids": tag_ids,
                    }
                )

        print(
            f"[MovieLensTagsDataset] Exemples carregados: {len(self.examples)} "
            f"(a partir de {tags_jsonl_path})"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def movielens_tags_collate_fn(
    batch: List[Dict[str, Any]],
    num_tags: int,
):
    """
    Collate para tarefa de tags.
    Retorna:
      - movie_indices: (B,)
      - tag_labels: (B, num_tags) multi-hot
    """
    batch_size = len(batch)
    movie_indices = torch.zeros(batch_size, dtype=torch.long)
    tag_labels = torch.zeros((batch_size, num_tags), dtype=torch.float32)

    for i, item in enumerate(batch):
        movie_indices[i] = item["movie_idx"]
        for t in item["tag_ids"]:
            if 0 <= t < num_tags:
                tag_labels[i, t] = 1.0

    return {
        "movie_indices": movie_indices,
        "tag_labels": tag_labels,
    }
