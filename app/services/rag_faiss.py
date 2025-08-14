# app/services/rag_faiss.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.services.rag_preprocess import CLEAN_CORPUS, _ensure_dirs, load_raw_jsonl

# --- пути для индекса и векторов ---
ROOT_DIR = Path(__file__).resolve().parents[1]
EMB_DIR = ROOT_DIR / "embeddings"
FAISS_INDEX_PATH = EMB_DIR / "few_shot_index.faiss"
VECTORS_PATH = EMB_DIR / "few_shot_vectors.npy"

# --- путь к локальной модели ---
LOCAL_MODEL_PATH = '/home/kusapochka/jobgpt/models/all-MiniLM-L6-v2'

# --- работа с корпусом ---
def load_clean_corpus(path: Path = CLEAN_CORPUS) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"чистый корпус не найден: {path}")
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# --- FAISS и эмбеддинги ---
def build_faiss_index(model_path: str | Path = LOCAL_MODEL_PATH) -> None:
    _ensure_dirs()
    corpus = load_clean_corpus()
    vacancies = [rec["vacancy"] for rec in corpus]

    model = SentenceTransformer(str(model_path))
    vectors = model.encode(vacancies, show_progress_bar=True, convert_to_numpy=True)

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    faiss.write_index(index, str(FAISS_INDEX_PATH))
    np.save(VECTORS_PATH, vectors)
    print(f"FAISS-индекс сохранён в: {FAISS_INDEX_PATH}")

def load_faiss_index(model_path: str | Path = LOCAL_MODEL_PATH) -> Tuple[faiss.Index, List[Dict], SentenceTransformer]:
    corpus = load_clean_corpus()
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    model = SentenceTransformer(str(model_path))
    return index, corpus, model

def search_few_shot(query: str, k: int = 3) -> List[Dict]:
    index, corpus, model = load_faiss_index()
    query_vec = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, k)
    results = [corpus[i] for i in I[0]]
    return results

# --- CLI для теста ---
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("использование:")
        print("  python -m app.services.rag_faiss build")
        print('  python -m app.services.rag_faiss search "текст вакансии"')
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "build":
        build_faiss_index()

    elif cmd == "search":
        if len(sys.argv) < 3:
            print("укажите текст вакансии для поиска")
            sys.exit(1)
        query = sys.argv[2]
        results = search_few_shot(query, k=3)
        for i, r in enumerate(results, 1):
            print(f"\nрезультат {i}:")
            print(f"vacancy: {r['vacancy'][:100]}...")
            print(f"response: {r['response'][:100]}...")

    else:
        print(f"неизвестная команда: {cmd}")