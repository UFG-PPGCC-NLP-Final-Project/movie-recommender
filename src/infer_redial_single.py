import json
import csv
from pathlib import Path

import torch
from torch import nn
from transformers import BertTokenizer

from model_bert_rnn import BertRNNRecommender


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 10

# Caminhos serão encontrados automaticamente pelas funções abaixo
TEST_PATH = None
VOCAB_PATH = None
MOVIES_CSV = None


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
    
    # CSV opcional
    possible_csv_paths = [
        Path("redial/movies_with_mentions.csv"),
        Path("datasets/redial/movies_with_mentions.csv"),
        Path("../redial/movies_with_mentions.csv"),
        Path("../datasets/redial/movies_with_mentions.csv"),
        Path(__file__).parent.parent / "redial" / "movies_with_mentions.csv",
        Path(__file__).parent.parent / "datasets" / "redial" / "movies_with_mentions.csv",
    ]
    
    movies_csv = None
    for csv_path in possible_csv_paths:
        if csv_path.exists():
            movies_csv = csv_path.resolve()
            break
    
    return test_path, vocab_path, movies_csv


def load_processed_example(idx: int, test_path: Path):
    """Carrega o exemplo idx do test.jsonl processado."""
    with test_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                obj = json.loads(line)
                return obj
    raise IndexError(f"Exemplo #{idx} não encontrado em {test_path}")


def load_movie_vocab(vocab_path: Path):
    """Carrega mapeamento movieIdOriginal → índice interno e cria inverso."""
    with vocab_path.open("r", encoding="utf-8") as f:
        movie_id_to_index = json.load(f)
    # chaves vêm como str se salvou assim, garante int
    movie_id_to_index = {int(k): int(v) for k, v in movie_id_to_index.items()}
    index_to_movie_id = {v: k for k, v in movie_id_to_index.items()}
    return movie_id_to_index, index_to_movie_id


def load_movie_titles(movies_csv: Path = None):
    """
    Tenta carregar um dicionário movieIdOriginal → título
    a partir do movies_with_mentions.csv.
    """
    titles = {}
    if movies_csv is None or not movies_csv.exists():
        if movies_csv:
            print(f"Aviso: {movies_csv} não encontrado. Vou mostrar só IDs.")
        return titles

    with movies_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # tentativa genérica: assume 1ª coluna = id, 2ª = título
        for row in reader:
            if len(row) < 2:
                continue
            try:
                mid = int(row[0])
            except ValueError:
                continue
            title = row[1]
            titles[mid] = title
    return titles


def prepare_tensors_from_example(example, tokenizer, max_len, num_movies):
    """
    Constrói tensores de entrada (BERT + RNN) a partir de um exemplo processado.
    """
    bert_text = example["bert_text"]
    rnn_movies = example["rnn_movies"]  # índices internos (0..num_movies-1)

    # ----- BERT -----
    enc = tokenizer(
        bert_text,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    input_ids = enc["input_ids"]          # (1, L)
    attention_mask = enc["attention_mask"]
    token_type_ids = enc["token_type_ids"]

    # ----- RNN -----
    # fazemos o mesmo shift de +1 usado no collate_fn (0 = PAD)
    if len(rnn_movies) == 0:
        # fallback, caso algum exemplo bizarro apareça
        rnn_input = torch.zeros((1, 1), dtype=torch.long)
        rnn_mask = torch.zeros((1, 1), dtype=torch.bool)
    else:
        shifted = [m + 1 for m in rnn_movies]
        rnn_input = torch.tensor(shifted, dtype=torch.long).unsqueeze(0)  # (1, T)
        rnn_mask = torch.ones_like(rnn_input, dtype=torch.bool)           # (1, T)

    # labels não são necessários na inferência
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "rnn_input": rnn_input,
        "rnn_mask": rnn_mask,
    }


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


def main():
    # -------- 1) Carrega checkpoint --------
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
        print("  3. Especifique o caminho manualmente no código")
        print("=" * 60)
        return
    
    print(f"Checkpoint encontrado: {ckpt_path}\n")
    
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

    print(f"Carregando modelo: {model_name}")
    print(f"num_movies={num_movies}, rnn_vocab_size={rnn_vocab_size}, max_len={max_len}")

    # -------- 2) Reconstrói modelo e tokenizer --------
    model = BertRNNRecommender(
        model_name=model_name,
        num_movies=num_movies,
        rnn_vocab_size=rnn_vocab_size,
        freeze_bert=False,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(model_name)

    # -------- 2.5) Encontra arquivos de dados --------
    test_path, vocab_path, movies_csv = find_data_files()
    
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
    print(f"  - movie_id_to_index.json: {vocab_path}")
    if movies_csv:
        print(f"  - movies_with_mentions.csv: {movies_csv}")
    print()

    # vocabulário e títulos (opcional)
    movie_id_to_index, index_to_movie_id = load_movie_vocab(vocab_path)
    movie_titles = load_movie_titles(movies_csv)

    # -------- 3) Escolhe um exemplo do test.jsonl --------
    example_idx = 0  # mude esse índice para ver outros diálogos
    example = load_processed_example(example_idx, test_path)

    print(f"\n=== Exemplo #{example_idx} do test.jsonl ===")
    print("Texto (bert_text):")
    print(example["bert_text"])
    print("\nFilmes citados pelo usuário (rnn_movies, índices internos):")
    print(example["rnn_movies"])

    # -------- 4) Prepara tensores --------
    batch = prepare_tensors_from_example(example, tokenizer, max_len, num_movies)

    # move para device
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    token_type_ids = batch["token_type_ids"].to(DEVICE)
    rnn_input = batch["rnn_input"].to(DEVICE)
    rnn_mask = batch["rnn_mask"].to(DEVICE)

    # -------- 5) Inferência --------
    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            rnn_input=rnn_input,
            rnn_mask=rnn_mask,
        )
        probs = torch.sigmoid(logits)[0]  # (num_movies,)

    # opcional: não recomendar filmes já citados na sequência do usuário
    seen_indices = set(example["rnn_movies"])
    mask = torch.ones_like(probs, dtype=torch.bool)
    for idx in seen_indices:
        if 0 <= idx < num_movies:
            mask[idx] = False
    probs_masked = probs.clone()
    probs_masked[~mask] = 0.0

    # -------- 6) Top-K recomendações --------
    topk = min(TOP_K, num_movies)
    top_values, top_indices = torch.topk(probs_masked, k=topk)

    print(f"\nTop-{topk} recomendações (índice interno → movieId → título):")
    for score, idx in zip(top_values.tolist(), top_indices.tolist()):
        movie_id = index_to_movie_id.get(idx, None)
        title = movie_titles.get(movie_id, "(título desconhecido)") if movie_id is not None else "(movieId desconhecido)"

        print(f"- score={score:.4f} | idx={idx} | movieId={movie_id} | {title}")


if __name__ == "__main__":
    main()
