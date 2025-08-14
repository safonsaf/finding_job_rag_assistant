from fastapi import FastAPI
from app.api.endpoints.upload_resume import router as upload_router

app = FastAPI()

app.include_router(upload_router)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "jobgpt is running!"}