# eval_redial_multitask.py
#
# Avalia o modelo multitask (BERT + RNN + MovieLens tags)
# em cima do TEST do ReDial, com:
#   - BCE Loss
#   - Precision@10
#   - Recall@10
#   - nDCG@10

from pathlib import Path
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from redial_dataset_eval import RedialEvalDataset, redial_eval_collate_fn
from model_bert_rnn_multitask import BertRNNMultiTaskRecommender

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 10

# Ajuste esses caminhos se necessário
REDIAL_DIR = Path("../datasets/redial/processed")
MOVIELENS_DIR = Path("../datasets/movielens/processed")

TEST_PATH = REDIAL_DIR / "test.jsonl"
MOVIE_VOCAB_PATH = REDIAL_DIR / "movie_id_to_index.json"
TAG_VOCAB_PATH = MOVIELENS_DIR / "tag_to_index.json"

# use o checkpoint da última época multitask (ou o que você quiser comparar)
CKPT_PATH = Path("checkpoints/multitask/multitask_epoch12.pth")



def load_movie_vocab():
    with MOVIE_VOCAB_PATH.open("r", encoding="utf-8") as f:
        movie_id_to_index = json.load(f)
    movie_id_to_index = {int(k): int(v) for k, v in movie_id_to_index.items()}
    num_movies = len(movie_id_to_index)
    return movie_id_to_index, num_movies


def load_tag_vocab():
    with TAG_VOCAB_PATH.open("r", encoding="utf-8") as f:
        tag_to_index = json.load(f)
    num_tags = len(tag_to_index)
    return tag_to_index, num_tags


def dcg_at_k(relevances, k):
    """relevances: lista de 0/1 em ordem de ranking."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        if relevances[i] > 0:
            # posição começa em 1 (i+1), mas fórmula usa log2(i+2)
            dcg += float(relevances[i]) / float(torch.log2(torch.tensor(i + 2.0)))
    return dcg


def main():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint multitask não encontrado: {CKPT_PATH}")

    _, num_movies = load_movie_vocab()
    _, num_tags = load_tag_vocab()

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model_name = ckpt.get("model_name", "bert-base-uncased")
    max_len = ckpt.get("max_len", 128)

    print(f"Modelo multitask: {model_name}")
    print(f"num_movies={num_movies}, num_tags={num_tags}, max_len={max_len}")

    # Reconstrói modelo multitask
    model = BertRNNMultiTaskRecommender(
        model_name=model_name,
        num_movies=num_movies,
        num_tags=num_tags,
        freeze_bert=False,
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Dataset + DataLoader para TEST do ReDial
    tokenizer = BertTokenizer.from_pretrained(model_name)
    test_dataset = RedialEvalDataset(TEST_PATH)

    def collate(batch):
        return redial_eval_collate_fn(
            batch=batch,
            tokenizer=tokenizer,
            max_len=max_len,
            num_movies=num_movies,
        )

    loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate,
    )

    criterion = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    n_batches = 0

    sum_tp = 0
    sum_pred = 0
    sum_true = 0

    sum_ndcg = 0.0
    n_examples = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            rnn_input = batch["rnn_input"].to(DEVICE)
            rnn_mask = batch["rnn_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            rnn_movies_list = batch["rnn_movies_list"]

            # usamos só a cabeça ReDial do modelo multitask
            logits = model.forward_redial(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                rnn_input=rnn_input,
                rnn_mask=rnn_mask,
            )

            loss = criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits)
            B = probs.size(0)

            for i in range(B):
                true_idx = (labels[i] > 0.5).nonzero(as_tuple=False).view(-1)
                if true_idx.numel() == 0:
                    continue

                true_set = set(true_idx.tolist())
                p = probs[i].clone()

                # zera filmes já vistos pelo usuário
                for idx in rnn_movies_list[i]:
                    if 0 <= idx < num_movies:
                        p[idx] = 0.0

                k = min(TOP_K, num_movies)
                _, top_idx = torch.topk(p, k=k)
                pred_set = set(top_idx.tolist())

                tp = len(true_set & pred_set)
                sum_tp += tp
                sum_pred += k
                sum_true += len(true_set)

                # nDCG
                rels = [1.0 if idx in true_set else 0.0 for idx in top_idx.tolist()]
                dcg = dcg_at_k(rels, k)
                ideal_rels = [1.0] * min(len(true_set), k)
                ideal_dcg = dcg_at_k(ideal_rels, k)
                ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

                sum_ndcg += ndcg
                n_examples += 1

    avg_loss = total_loss / max(1, n_batches)
    precision = sum_tp / max(1, sum_pred)
    recall = sum_tp / max(1, sum_true)
    mean_ndcg = sum_ndcg / max(1, n_examples)

    print("\n===== RESULTADOS (modelo multitask) =====")
    print(f"BCE Loss média: {avg_loss:.6f}")
    print(f"Precision@{TOP_K}: {precision:.6f}")
    print(f"Recall@{TOP_K}:    {recall:.6f}")
    print(f"nDCG@{TOP_K}:      {mean_ndcg:.6f}")


if __name__ == "__main__":
    main()
