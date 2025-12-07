# train_multitask.py
#
# Treino multi-task:
#   - ReDial (recomendação de filmes)
#   - MovieLens (predição de tags de filme)

from pathlib import Path
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from redial_dataset import RedialDataset, redial_collate_fn
from movielens_tags_dataset import MovieLensTagsDataset, movielens_tags_collate_fn
from model_bert_rnn_multitask import BertRNNMultiTaskRecommender

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ===== Config =====
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_REDIR = 8
BATCH_TAGS = 256
EPOCHS = 30
LR = 1e-5
LAMBDA_TAGS = 1.0
FREEZE_BERT = False

# Caminhos
REDIAL_DIR = Path("../datasets/redial/processed")
MOVIELENS_DIR = Path("../datasets/movielens/processed")
CKPT_DIR = Path("checkpoints/multitask")


def create_redial_loader(num_movies, tokenizer):
    train_path = REDIAL_DIR / "train.jsonl"
    dataset = RedialDataset(train_path)

    def collate(batch):
        return redial_collate_fn(
            batch=batch,
            tokenizer=tokenizer,
            max_len=MAX_LEN,
            num_movies=num_movies,
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_REDIR,
        shuffle=True,
        collate_fn=collate,
    )
    return loader


def create_tags_loader(movie_id_to_index_path, num_tags):
    tags_jsonl = MOVIELENS_DIR / "movielens_tags.jsonl"
    dataset = MovieLensTagsDataset(
        tags_jsonl_path=tags_jsonl,
        movie_id_to_index_path=movie_id_to_index_path,
    )

    def collate(batch):
        return movielens_tags_collate_fn(
            batch=batch,
            num_tags=num_tags,
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_TAGS,
        shuffle=True,
        collate_fn=collate,
    )
    return loader


def main():
    CKPT_DIR.mkdir(exist_ok=True)

    # ----- carrega vocabulários -----
    movie_vocab_path = REDIAL_DIR / "movie_id_to_index.json"
    with movie_vocab_path.open("r", encoding="utf-8") as f:
        movie_id_to_index = json.load(f)
    num_movies = len(movie_id_to_index)

    tag_vocab_path = MOVIELENS_DIR / "tag_to_index.json"
    with tag_vocab_path.open("r", encoding="utf-8") as f:
        tag_to_index = json.load(f)
    num_tags = len(tag_to_index)

    print(f"num_movies: {num_movies}")
    print(f"num_tags:   {num_tags}")

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    redial_loader = create_redial_loader(num_movies, tokenizer)
    tags_loader = create_tags_loader(movie_vocab_path, num_tags)

    # ----- modelo multitask -----
    model = BertRNNMultiTaskRecommender(
        model_name=MODEL_NAME,
        num_movies=num_movies,
        num_tags=num_tags,
        freeze_bert=FREEZE_BERT,
    ).to(DEVICE)

    criterion_redial = nn.BCEWithLogitsLoss()
    criterion_tags = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_redial_loss = 0.0
        total_tags_loss = 0.0
        n_steps = 0

        redial_iter = iter(redial_loader)
        tags_iter = iter(tags_loader)

        # número de passos = menor loader, para evitar exaustão de um lado só
        steps_per_epoch = min(len(redial_loader), len(tags_loader))

        for step in range(steps_per_epoch):
            try:
                redial_batch = next(redial_iter)
            except StopIteration:
                redial_iter = iter(redial_loader)
                redial_batch = next(redial_iter)

            try:
                tags_batch = next(tags_iter)
            except StopIteration:
                tags_iter = iter(tags_loader)
                tags_batch = next(tags_iter)

            # ----- ReDial loss -----
            input_ids = redial_batch["input_ids"].to(DEVICE)
            attention_mask = redial_batch["attention_mask"].to(DEVICE)
            token_type_ids = redial_batch["token_type_ids"].to(DEVICE)
            rnn_input = redial_batch["rnn_input"].to(DEVICE)
            rnn_mask = redial_batch["rnn_mask"].to(DEVICE)
            labels_movies = redial_batch["labels"].to(DEVICE)

            logits_movies = model.forward_redial(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                rnn_input=rnn_input,
                rnn_mask=rnn_mask,
            )
            loss_redial = criterion_redial(logits_movies, labels_movies)

            # ----- Tags loss -----
            movie_indices = tags_batch["movie_indices"].to(DEVICE)
            tag_labels = tags_batch["tag_labels"].to(DEVICE)

            logits_tags = model.forward_tags(movie_indices)
            loss_tags = criterion_tags(logits_tags, tag_labels)

            # ----- Loss total -----
            loss = loss_redial + LAMBDA_TAGS * loss_tags

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_redial_loss += loss_redial.item()
            total_tags_loss += loss_tags.item()
            n_steps += 1

            if (step + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1}/{EPOCHS} Step {step+1}/{steps_per_epoch} "
                    f"- L_redial: {loss_redial.item():.4f} "
                    f"- L_tags: {loss_tags.item():.4f}"
                )

        avg_redial = total_redial_loss / max(1, n_steps)
        avg_tags = total_tags_loss / max(1, n_steps)
        print(
            f"Epoch {epoch+1}/{EPOCHS} - "
            f"Train L_redial: {avg_redial:.4f} | Train L_tags: {avg_tags:.4f}"
        )

        # salva checkpoint de época
        ckpt_path = CKPT_DIR / f"multitask_epoch{epoch+1}.pth"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_movies": num_movies,
                "num_tags": num_tags,
                "model_name": MODEL_NAME,
                "max_len": MAX_LEN,
            },
            ckpt_path,
        )
        print(f"Checkpoint salvo em: {ckpt_path}")

    print("Treino multitask concluído.")


if __name__ == "__main__":
    main()
