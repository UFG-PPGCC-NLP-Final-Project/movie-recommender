# eval_redial_metrics.py
#
# Avaliação completa de:
#   - Precision@10
#   - Recall@10
#   - nDCG@10
#
# Usando somente arquivos novos, sem alterar nada do treino.

from pathlib import Path
import json
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from redial_dataset_eval import RedialEvalDataset, redial_eval_collate_fn
from model_bert_rnn import BertRNNRecommender

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 10

# Caminhos serão encontrados automaticamente pelas funções abaixo
TEST_PATH = None
VOCAB_PATH = None


# ------------- Helpers ---------------------

def find_checkpoint():
    """Tenta encontrar o checkpoint em vários locais possíveis."""
    possible_paths = [
        Path("checkpoints/best_model.pth"),
        Path("datasets/checkpoints/best_model.pth"),
        Path("../checkpoints/best_model.pth"),
        Path("../datasets/checkpoints/best_model.pth"),
        Path(__file__).parent.parent / "checkpoints" / "best_model.pth",
        Path(__file__).parent.parent / "datasets" / "checkpoints" / "best_model.pth",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path.resolve()
    
    return None


def find_data_files():
    """Tenta encontrar os arquivos de dados em vários locais possíveis."""
    possible_base_dirs = [
        Path("redial/processed"),
        Path("datasets/redial/processed"),
        Path("../redial/processed"),
        Path("../datasets/redial/processed"),
        Path(__file__).parent.parent / "redial" / "processed",
        Path(__file__).parent.parent / "datasets" / "redial" / "processed",
    ]
    
    test_path = None
    vocab_path = None
    
    for base_dir in possible_base_dirs:
        test_file = base_dir / "test.jsonl"
        vocab_file = base_dir / "movie_id_to_index.json"
        
        if test_file.exists() and vocab_file.exists():
            test_path = test_file.resolve()
            vocab_path = vocab_file.resolve()
            break
    
    return test_path, vocab_path


def load_vocab(vocab_path: Path):
    """Carrega o vocabulário de filmes."""
    with vocab_path.open("r", encoding="utf-8") as f:
        movie_id_to_index = json.load(f)
    movie_id_to_index = {int(k): int(v) for k, v in movie_id_to_index.items()}
    return movie_id_to_index, len(movie_id_to_index)


def dcg_at_k(relevances, k):
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        if relevances[i] > 0:
            # garantimos coerência sem depender de .item()
            dcg += float(relevances[i]) / float(torch.log2(torch.tensor(i + 2.0)))
    return dcg




# ------------- Avaliação ---------------------

def main():
    # -------- 1) Encontra checkpoint --------
    ckpt_path = find_checkpoint()
    
    if ckpt_path is None:
        print("=" * 60)
        print("ERRO: Checkpoint não encontrado!")
        print("=" * 60)
        print("\nO script procurou nos seguintes locais:")
        possible_paths = [
            "checkpoints/best_model.pth",
            "datasets/checkpoints/best_model.pth",
            "../checkpoints/best_model.pth",
            "../datasets/checkpoints/best_model.pth",
        ]
        for path in possible_paths:
            print(f"  - {path}")
        print("\nSugestões:")
        print("  1. Execute primeiro o treinamento (opção 3 do menu)")
        print("  2. Verifique se o checkpoint foi salvo em outro local")
        print("=" * 60)
        return
    
    print(f"Checkpoint encontrado: {ckpt_path}\n")
    
    # -------- 2) Encontra arquivos de dados --------
    test_path, vocab_path = find_data_files()
    
    if test_path is None or vocab_path is None:
        print("=" * 60)
        print("ERRO: Arquivos de dados não encontrados!")
        print("=" * 60)
        print("\nO script procurou nos seguintes locais:")
        possible_paths = [
            "redial/processed/test.jsonl",
            "datasets/redial/processed/test.jsonl",
            "redial/processed/movie_id_to_index.json",
            "datasets/redial/processed/movie_id_to_index.json",
        ]
        for path in possible_paths:
            print(f"  - {path}")
        print("\nSugestões:")
        print("  1. Execute primeiro a preparação de dados (opção 1 do menu)")
        print("  2. Verifique se os dados foram processados corretamente")
        print("=" * 60)
        return
    
    print(f"Arquivos de dados encontrados:")
    print(f"  - test.jsonl: {test_path}")
    print(f"  - movie_id_to_index.json: {vocab_path}\n")

    # Carrega checkpoint
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
    except Exception as e:
        print(f"ERRO ao carregar checkpoint: {e}")
        print("Verifique se o arquivo está corrompido ou em formato incompatível.")
        return
    num_movies = ckpt["num_movies"]
    rnn_vocab_size = ckpt["rnn_vocab_size"]
    model_name = ckpt.get("model_name", "bert-base-uncased")
    max_len = ckpt.get("max_len", 128)

    print(f"Modelo: {model_name}")
    print(f"num_movies={num_movies}, rnn_vocab_size={rnn_vocab_size}, max_len={max_len}")

    # Reconstrói modelo
    model = BertRNNRecommender(
        model_name=model_name,
        num_movies=num_movies,
        rnn_vocab_size=rnn_vocab_size,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Dataset de avaliação
    tokenizer = BertTokenizer.from_pretrained(model_name)
    test_dataset = RedialEvalDataset(test_path)

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

    # Métricas acumuladas
    sum_tp = 0
    sum_pred = 0
    sum_true = 0

    sum_ndcg = 0.0
    n_examples = 0

    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0

    # Loop de avaliação
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            rnn_input = batch["rnn_input"].to(DEVICE)
            rnn_mask = batch["rnn_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                rnn_input=rnn_input,
                rnn_mask=rnn_mask,
            )

            # BCE loss (informativo)
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

                # Remove filmes já citados
                for idx in batch["rnn_movies_list"][i]:
                    if 0 <= idx < num_movies:
                        p[idx] = 0

                k = min(TOP_K, num_movies)
                _, top_idx = torch.topk(p, k=k)

                pred_set = set(top_idx.tolist())

                tp = len(true_set & pred_set)
                sum_tp += tp
                sum_pred += k
                sum_true += len(true_set)

                # nDCG
                rels = []
                for idx in top_idx.tolist():
                    rels.append(1.0 if idx in true_set else 0.0)

                dcg = dcg_at_k(torch.tensor(rels, dtype=torch.float32), k)
                ideal = dcg_at_k(torch.ones(min(len(true_set), k)), k)
                ndcg = dcg / ideal if ideal > 0 else 0.0

                sum_ndcg += ndcg
                n_examples += 1

    # Métricas finais
    precision = sum_tp / sum_pred
    recall = sum_tp / sum_true
    ndcg = sum_ndcg / n_examples
    avg_loss = total_loss / n_batches

    print("\n===== RESULTADOS =====")
    print(f"BCE Loss média: {avg_loss:.6f}")
    print(f"Precision@{TOP_K}: {precision:.6f}")
    print(f"Recall@{TOP_K}:    {recall:.6f}")
    print(f"nDCG@{TOP_K}:      {ndcg:.6f}")


if __name__ == "__main__":
    main()
