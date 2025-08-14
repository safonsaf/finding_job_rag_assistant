# app/services/rag_preprocess.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# пути
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EMB_DIR = ROOT_DIR / "embeddings"

SRC_JSONL = DATA_DIR / "few_shot_examples_cleaned.jsonl"
CLEAN_CORPUS = EMB_DIR / "few_shot_corpus.jsonl"

# --- утилиты ---
def _ensure_dirs() -> None:
    EMB_DIR.mkdir(parents=True, exist_ok=True)

def _normalize_ws(text: str) -> str:
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _valid_record(obj: Dict) -> bool:
    if not isinstance(obj, dict):
        return False
    if "vacancy" not in obj or "response" not in obj:
        return False
    if not isinstance(obj["vacancy"], str) or not isinstance(obj["response"], str):
        return False
    if _normalize_ws(obj["vacancy"]) == "" or _normalize_ws(obj["response"]) == "":
        return False
    return True

# --- подготовка корпуса ---
def load_raw_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    if not path.exists():
        raise FileNotFoundError(f"файл не найден: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append(obj)
    return records

def prepare_corpus(raw: List[Dict]) -> Tuple[List[Dict], List[int]]:
    cleaned: List[Dict] = []
    kept: List[int] = []

    for i, obj in enumerate(raw):
        if not _valid_record(obj):
            continue
        vac = _normalize_ws(obj["vacancy"])
        rsp = _normalize_ws(obj["response"])
        if not vac or not rsp:
            continue
        cleaned.append({"vacancy": vac, "response": rsp})
        kept.append(i)

    return cleaned, kept

def save_corpus_jsonl(corpus: List[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in corpus:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def build_clean_corpus(
    src_path: Path = SRC_JSONL,
    out_path: Path = CLEAN_CORPUS
) -> Dict[str, int]:
    _ensure_dirs()
    raw = load_raw_jsonl(src_path)
    cleaned, kept_idx = prepare_corpus(raw)
    save_corpus_jsonl(cleaned, out_path)
    return {
        "total": len(raw),
        "kept": len(cleaned),
        "dropped": len(raw) - len(cleaned),
    }