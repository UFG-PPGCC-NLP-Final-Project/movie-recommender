import json
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Set, Optional

REDIAL_DIR = Path("../datasets/redial")
TRAIN_FILE = REDIAL_DIR / "train_data.jsonl"
TEST_FILE  = REDIAL_DIR / "test_data.jsonl"

MOVIE_REGEX = re.compile(r"@(\d+)")


def iter_jsonl(path: Path):
    """Itera sobre um arquivo .jsonl, linha por linha."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_movie_ids_from_text(text: str) -> List[int]:
    """Extrai IDs de filmes do texto (@12345 → 12345)."""
    return [int(m) for m in MOVIE_REGEX.findall(text)]


def replace_movie_ids_with_placeholder(text: str) -> str:
    """Substitui @12345 por @ (placeholder) para entrada do BERT."""
    return MOVIE_REGEX.sub("@", text)


def parse_dialogue(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Constrói um exemplo a partir de um diálogo no formato JSONL.

    - bert_text: falas do iniciador, concatenadas com [SEP]
    - initiator_movie_ids: IDs de filmes citados pelo iniciador (para RNN)
    - label_movie_ids: IDs de filmes recomendados pelo respondente (suggested == 1)
    """
    initiator_id = obj.get("initiatorWorkerId")
    respondent_id = obj.get("respondentWorkerId")
    messages = obj.get("messages", [])

    if initiator_id is None or respondent_id is None or not messages:
        return None

    initiator_texts: List[str] = []
    initiator_movie_ids: List[int] = []

    # 1) texto + filmes do iniciador
    for msg in messages:
        sender = msg.get("senderWorkerId")
        text = msg.get("text", "")

        if sender == initiator_id:
            clean_text = replace_movie_ids_with_placeholder(text)
            initiator_texts.append(clean_text)

            movie_ids = extract_movie_ids_from_text(text)
            initiator_movie_ids.extend(movie_ids)

    # 2) filmes recomendados pelo respondente (labels)
    label_movie_ids: List[int] = []
    resp_q = obj.get("respondentQuestions", {}) or {}
    for mid_str, info in resp_q.items():
        try:
            mid = int(mid_str)
        except ValueError:
            continue
        if info.get("suggested", 0) == 1:
            label_movie_ids.append(mid)

    # descarta diálogos sem texto do initiator ou sem recomendações
    if not initiator_texts or not label_movie_ids:
        return None

    bert_text = " [SEP] ".join(initiator_texts)

    return {
        "bert_text": bert_text,
        "initiator_movie_ids": sorted(set(initiator_movie_ids)),
        "label_movie_ids": sorted(set(label_movie_ids)),
    }


def build_movie_vocab(examples: List[Dict[str, Any]]) -> Dict[int, int]:
    """movie_id → índice contínuo."""
    movie_ids: Set[int] = set()
    for ex in examples:
        movie_ids.update(ex["initiator_movie_ids"])
        movie_ids.update(ex["label_movie_ids"])

    sorted_ids = sorted(movie_ids)
    return {mid: idx for idx, mid in enumerate(sorted_ids)}


def map_examples_to_indices(
    examples: List[Dict[str, Any]],
    movie_id_to_index: Dict[int, int],
) -> List[Dict[str, Any]]:
    """Converte IDs de filmes para índices internos."""
    mapped = []
    for ex in examples:
        rnn_indices = [movie_id_to_index[mid] for mid in ex["initiator_movie_ids"]
                       if mid in movie_id_to_index]
        label_indices = [movie_id_to_index[mid] for mid in ex["label_movie_ids"]
                         if mid in movie_id_to_index]

        if not rnn_indices or not label_indices:
            continue

        mapped.append({
            "bert_text": ex["bert_text"],
            "rnn_movies": rnn_indices,
            "labels": label_indices,
        })
    return mapped


def save_jsonl(path: Path, data: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for obj in data:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    print("Lendo train_data.jsonl...")
    train_raw = [obj for obj in iter_jsonl(TRAIN_FILE)]
    print("Lendo test_data.jsonl...")
    test_raw  = [obj for obj in iter_jsonl(TEST_FILE)]

    print(f"Diálogos brutos: train={len(train_raw)}, test={len(test_raw)}")

    def build_examples(raw_list):
        examples = []
        for d in raw_list:
            ex = parse_dialogue(d)
            if ex is not None:
                examples.append(ex)
        return examples

    train_examples = build_examples(train_raw)
    test_examples  = build_examples(test_raw)

    print(f"Exemplos úteis: train={len(train_examples)}, test={len(test_examples)}")

    # cria um val split a partir do train (80/20)
    random.shuffle(train_examples)
    split = int(0.8 * len(train_examples))
    train_split = train_examples[:split]
    valid_split = train_examples[split:]

    print(f"Split: train={len(train_split)}, valid={len(valid_split)}")

    # vocabulário de filmes global
    all_examples = train_split + valid_split + test_examples
    movie_id_to_index = build_movie_vocab(all_examples)
    print(f"Total de filmes únicos: {len(movie_id_to_index)}")

    # mapeia IDs → índices
    train_mapped = map_examples_to_indices(train_split, movie_id_to_index)
    valid_mapped = map_examples_to_indices(valid_split, movie_id_to_index)
    test_mapped  = map_examples_to_indices(test_examples,  movie_id_to_index)

    print(f"Após filtragem: train={len(train_mapped)}, valid={len(valid_mapped)}, test={len(test_mapped)}")

    out_dir = REDIAL_DIR / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(out_dir / "train.jsonl", train_mapped)
    save_jsonl(out_dir / "valid.jsonl", valid_mapped)
    save_jsonl(out_dir / "test.jsonl",  test_mapped)

    with (out_dir / "movie_id_to_index.json").open("w", encoding="utf-8") as f:
        json.dump(movie_id_to_index, f, ensure_ascii=False, indent=2)

    print("Concluído. Arquivos salvos em:", out_dir)


if __name__ == "__main__":
    main()
