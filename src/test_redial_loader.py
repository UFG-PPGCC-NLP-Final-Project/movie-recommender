from pathlib import Path
import json

from torch.utils.data import DataLoader
from transformers import BertTokenizer

from redial_dataset import RedialDataset, redial_collate_fn

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 4

def main():
    base_dir = Path("../datasets/redial/processed")
    train_path = base_dir / "train.jsonl"
    vocab_path = base_dir / "movie_id_to_index.json"

    # carrega num_movies
    with vocab_path.open("r", encoding="utf-8") as f:
        movie_id_to_index = json.load(f)
    num_movies = len(movie_id_to_index)
    print("num_movies:", num_movies)

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    dataset = RedialDataset(train_path)

    def collate(batch):
        return redial_collate_fn(
            batch=batch,
            tokenizer=tokenizer,
            max_len=MAX_LEN,
            num_movies=num_movies,
        )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    batch = next(iter(loader))

    print("input_ids shape:", batch["input_ids"].shape)       # (B, L)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("token_type_ids shape:", batch["token_type_ids"].shape)
    print("rnn_input shape:", batch["rnn_input"].shape)       # (B, T_movies)
    print("rnn_mask shape:", batch["rnn_mask"].shape)
    print("labels shape:", batch["labels"].shape)             # (B, num_movies)
    print("rnn_vocab_size:", batch["rnn_vocab_size"])

if __name__ == "__main__":
    main()
