from pathlib import Path
from typing import List, Dict

from app.services.resume_parser import parse_resume
from app.services.rag_faiss import search_few_shot

# пути к файлам
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESUME_SHORT_PATH = DATA_DIR / "resume_short.txt"


def load_static_resume_summary() -> str:
    """читает постоянную выжимку из файла resume_short.txt"""
    if not RESUME_SHORT_PATH.exists():
        raise FileNotFoundError(f"файл с кратким резюме не найден: {RESUME_SHORT_PATH}")
    return RESUME_SHORT_PATH.read_text(encoding="utf-8").strip()


def format_few_shot_examples(examples: List[Dict], static_resume_summary: str) -> str:
    """форматирует few-shot примеры для промпта"""
    formatted = []
    for ex in examples:
        block = (
            f"Вакансия:\n{ex['vacancy']}\n\n"
            f"Краткая выжимка резюме:\n{static_resume_summary}\n\n"
            f"Отклик:\n{ex['response']}"
        )
        formatted.append(block)
    return "\n\n---\n\n".join(formatted)


def build_prompt(vacancy_text: str, resume_path: str) -> str:
    """
    собирает финальный промпт:
    1. краткое описание опыта из нового резюме
    2. few-shot примеры (с постоянной выжимкой резюме)
    3. инструкция LLM
    """
    # блок 1 — опыт из нового резюме
    new_resume_summary = parse_resume(resume_path)

    # блок 2 — few-shot примеры
    static_resume_summary = load_static_resume_summary()
    few_shot = search_few_shot(vacancy_text, k=3)
    few_shot_block = format_few_shot_examples(few_shot, static_resume_summary)

    # блок 3 — инструкция
    instruction = (
        "Используя краткое описание опыта кандидата, примеры откликов и текст вакансии, "
        "составь персонализированный отклик, который будет релевантен этой вакансии и основан на опыте кандидата."
    )

    # собираем финальный промпт
    prompt = (
        f"Краткое описание нового резюме:\n{new_resume_summary}\n\n"
        f"Примеры откликов:\n{few_shot_block}\n\n"
        f"Входная вакансия:\n{vacancy_text}\n\n"
        f"{instruction}"
    )

    return prompt