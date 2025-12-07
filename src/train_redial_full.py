from pathlib import Path
import json
import os
from time import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from redial_dataset import RedialDataset, redial_collate_fn
from model_bert_rnn import BertRNNRecommender

# Descrição: Treinamento completo do modelo BERT-RNN para o dataset ReDial.

# ==============================
# Configurações
# ==============================
MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-5
FREEZE_BERT = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def find_processed_dir():
    """Encontra o diretório de dados processados em vários locais possíveis."""
    possible_dirs = [
        Path("redial/processed"),
        Path("datasets/redial/processed"),
        Path("../redial/processed"),
        Path("../datasets/redial/processed"),
        Path(__file__).parent.parent / "redial" / "processed",
        Path(__file__).parent.parent / "datasets" / "redial" / "processed",
    ]
    
    for processed_dir in possible_dirs:
        vocab_file = processed_dir / "movie_id_to_index.json"
        train_file = processed_dir / "train.jsonl"
        valid_file = processed_dir / "valid.jsonl"
        
        if vocab_file.exists() and train_file.exists() and valid_file.exists():
            return processed_dir.resolve()
    
    return None


def create_dataloaders(base_dir: Path):
    # o que são dataloaders?
    # são objetos que carregam os dados do dataset em lotes (batches) para o treinamento do modelo
    # eles são usados para iterar sobre o dataset em lotes, e para isso eles precisam de um collate_fn
    # o collate_fn é uma função que coloca os dados em um formato que o modelo pode entender para o treinamento.

    train_path = base_dir / "train.jsonl"
    valid_path = base_dir / "valid.jsonl"
    vocab_path = base_dir / "movie_id_to_index.json"

    # num_movies 
    with vocab_path.open("r", encoding="utf-8") as f:
        movie_id_to_index = json.load(f)
    num_movies = len(movie_id_to_index)
    rnn_vocab_size = num_movies + 1  # por causa do shift no collate

    print("num_movies:", num_movies)
    print("rnn_vocab_size:", rnn_vocab_size)

    # tokenizer é um objeto que tokeniza os textos para o modeloBERT
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # train_dataset e valid_dataset são objetos que carregam os dados do dataset para o treinamento e validação
    train_dataset = RedialDataset(train_path)
    valid_dataset = RedialDataset(valid_path)

    def collate(batch):
        return redial_collate_fn(
            batch=batch,
            tokenizer=tokenizer,
            max_len=MAX_LEN,
            num_movies=num_movies,
        )

    # train_loader e valid_loader são objetos que carregam os dados do dataset para o treinamento e validação
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
    )

    return train_loader, valid_loader, num_movies, rnn_vocab_size


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        rnn_input = batch["rnn_input"].to(device)
        rnn_mask = batch["rnn_mask"].to(device)
        labels = batch["labels"].to(device)  # (B, num_movies)

        optimizer.zero_grad()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            rnn_input=rnn_input,
            rnn_mask=rnn_mask,
        )

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            rnn_input = batch["rnn_input"].to(device)
            rnn_mask = batch["rnn_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                rnn_input=rnn_input,
                rnn_mask=rnn_mask,
            )

            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)


def main():
    # -------- 1) Encontra diretório de dados processados --------
    base_dir = find_processed_dir()
    
    if base_dir is None:
        print("=" * 60)
        print("ERRO: Diretório de dados processados não encontrado!")
        print("=" * 60)
        print("\nO script procurou nos seguintes locais:")
        possible_paths = [
            "redial/processed/",
            "datasets/redial/processed/",
            "../redial/processed/",
            "../datasets/redial/processed/",
        ]
        for path in possible_paths:
            print(f"  - {path}")
        print("\nSugestões:")
        print("  1. Execute primeiro a preparação de dados (opção 1 do menu)")
        print("  2. Verifique se os arquivos foram processados corretamente")
        print("  3. Verifique se os arquivos estão no diretório correto")
        print("=" * 60)
        return
    
    print(f"Diretório de dados processados encontrado: {base_dir}\n")

    train_loader, valid_loader, num_movies, rnn_vocab_size = create_dataloaders(base_dir)

    model = BertRNNRecommender(
        model_name=MODEL_NAME,
        num_movies=num_movies,
        rnn_vocab_size=rnn_vocab_size,
        freeze_bert=FREEZE_BERT,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_valid_loss = float("inf")

    # Tenta salvar checkpoint em vários locais possíveis
    possible_ckpt_dirs = [
        Path("checkpoints"),
        Path("datasets/checkpoints"),
        Path(__file__).parent.parent / "checkpoints",
        Path(__file__).parent.parent / "datasets" / "checkpoints",
    ]
    
    ckpt_dir = None
    for ckpt_path in possible_ckpt_dirs:
        try:
            ckpt_path.mkdir(parents=True, exist_ok=True)
            ckpt_dir = ckpt_path
            break
        except:
            continue
    
    if ckpt_dir is None:
        # Fallback: cria no diretório atual
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(exist_ok=True)
    
    print(f"Checkpoints serão salvos em: {ckpt_dir}\n")

    for epoch in range(EPOCHS):
        initial_time = time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        valid_loss = evaluate(model, valid_loader, criterion)
        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"Train loss: {train_loss:.4f} | Valid loss: {valid_loss:.4f}")
        final_time = time()
        print(f"Tempo de execução do epoch {epoch+1}: {final_time - initial_time:.2f} segundos")
        # salva melhor modelo
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            ckpt_path = ckpt_dir / "best_model.pth"

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_movies": num_movies,
                    "rnn_vocab_size": rnn_vocab_size,
                    "model_name": MODEL_NAME,
                    "max_len": MAX_LEN,
                },
                ckpt_path,
            )
            print(f"  >> Novo melhor modelo salvo em {ckpt_path} "
                  f"(valid_loss={best_valid_loss:.4f})")

    print("Treino completo.")
    print("Melhor valid_loss:", best_valid_loss)


if __name__ == "__main__":
    initial_time = time()
    main()
    final_time = time()
    print(f"Tempo de execução: {final_time - initial_time:.2f} segundos")
