import csv
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Set, List

# Ajuste o caminho base se necessário
MOVIELENS_DIR = Path("../datasets/movielens")
TAGS_FILE = MOVIELENS_DIR / "tags.csv"

OUTPUT_DIR = MOVIELENS_DIR / "processed"
OUTPUT_TAGS_JSONL = OUTPUT_DIR / "movielens_tags.jsonl"
OUTPUT_TAG_VOCAB = OUTPUT_DIR / "tag_to_index.json"

# Hiperparâmetro: frequência mínima para manter uma tag no vocabulário
MIN_TAG_FREQ = 10  # pode ajustar depois


def normalize_tag(tag: str) -> str:
    """
    Normaliza uma tag:
      - strip
      - lower
    Se quiser, pode adicionar mais regras (remover pontuação, etc.).
    """
    return tag.strip().lower()


def main():
    if not TAGS_FILE.exists():
        raise FileNotFoundError(f"tags.csv não encontrado em: {TAGS_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Ler tags.csv e acumular:
    #   - contagem global de cada tag (para filtro por frequência)
    #   - conjunto de tags por movieId
    tag_counts: Counter = Counter()
    movie_to_tags: Dict[int, Set[str]] = defaultdict(set)

    print(f"Lendo {TAGS_FILE} ...")

    with TAGS_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                movie_id = int(row["movieId"])
            except ValueError:
                continue

            raw_tag = row.get("tag", "")
            tag = normalize_tag(raw_tag)

            if not tag:
                continue

            # conta frequência global (todas as ocorrências)
            tag_counts[tag] += 1

            # evita duplicatas por filme (guarda como set)
            movie_to_tags[movie_id].add(tag)

    print(f"Total de filmes com alguma tag: {len(movie_to_tags)}")
    print(f"Total de tags distintas (antes do filtro): {len(tag_counts)}")

    # 2) Construir vocabulário filtrando tags muito raras
    kept_tags: List[str] = [
        tag for tag, cnt in tag_counts.items() if cnt >= MIN_TAG_FREQ
    ]

    kept_tags.sort()
    tag_to_index = {tag: idx for idx, tag in enumerate(kept_tags)}

    print(f"Tags mantidas (freq >= {MIN_TAG_FREQ}): {len(kept_tags)}")

    # 3) Gerar movielens_tags.jsonl (movieId → lista de IDs de tags)
    num_movies_with_kept_tags = 0

    with OUTPUT_TAGS_JSONL.open("w", encoding="utf-8") as out_f:
        for movie_id, tags in movie_to_tags.items():
            tag_ids = [tag_to_index[t] for t in tags if t in tag_to_index]
            if not tag_ids:
                continue

            num_movies_with_kept_tags += 1

            obj = {
                "movieId": movie_id,
                "tag_ids": sorted(tag_ids),
            }
            out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # 4) Salvar vocabulário de tags
    with OUTPUT_TAG_VOCAB.open("w", encoding="utf-8") as f:
        json.dump(tag_to_index, f, ensure_ascii=False, indent=2)

    print(f"Filmes com ao menos uma tag mantida: {num_movies_with_kept_tags}")
    print(f"Arquivo de tags por filme salvo em: {OUTPUT_TAGS_JSONL}")
    print(f"Vocabulário de tags salvo em:      {OUTPUT_TAG_VOCAB}")


if __name__ == "__main__":
    main()
