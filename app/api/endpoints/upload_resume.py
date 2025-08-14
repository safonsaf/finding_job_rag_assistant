# app/api/endpoints/upload_resume.py

from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import uuid
from app.services.resume_parser import parse_resume

router = APIRouter()

@router.post("/upload_resume/")
async def upload_resume(file: UploadFile = File(...)):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Поддерживаются только PDF и DOCX")

    file_id = str(uuid.uuid4())
    temp_path = f"temp_{file_id}_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        summary = parse_resume(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")
    finally:
        try:
            import os
            os.remove(temp_path)
        except:
            pass

    return {"summary": summary}

