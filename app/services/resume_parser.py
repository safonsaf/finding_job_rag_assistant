import pdfplumber
import docx
from dotenv import load_dotenv
import os
from together import Together

load_dotenv()
API_KEY = os.getenv("TOGETHER_API_KEY")

client = Together(api_key=API_KEY)


def extract_text_from_pdf(file_path: str) -> str:
    """достаёт текст из pdf"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def extract_text_from_docx(file_path: str) -> str:
    """достаёт текст из docx"""
    text = ""
    doc = docx.Document(file_path)
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text


def generate_experience_summary(resume_text: str) -> str:
    """генерирует краткое описание опыта через LLM"""
    prompt = (
        "Вот текст резюме:\n"
        f"{resume_text}\n\n"
        "Составь краткое описание опыта кандидата в 4-5 предложениях: "
        "ключевые навыки, технологии, проекты. "
        "Не используй лишних деталей, только суть."
    )

    response = client.chat.completions.create(
        model="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300
    )

    return response.choices[0].message.content


def parse_resume(file_path: str) -> str:
    """главная функция: извлекает текст и делает краткое описание"""
    if file_path.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith(".docx"):
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Формат файла не поддерживается (только .pdf и .docx)")

    summary = generate_experience_summary(text)
    return summary

