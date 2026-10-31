"""FastAPI app. Run from the backend folder:  uvicorn app.main:app --reload
Interactive docs:  http://127.0.0.1:8000/docs
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import dataset_manager as dm
from app.pipeline import answer_question

app = FastAPI(title="AI Data Analyst API")

# Open CORS so the frontend dev server can call this. Restrict before deploying.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AskRequest(BaseModel):
    dataset: str
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/datasets")
def datasets():
    return dm.list_datasets()


@app.get("/datasets/{name}/preview")
def preview(name: str):
    try:
        return dm.get_schema(name)
    except dm.DatasetError as e:
        raise HTTPException(status_code=404, detail=str(e))


# Plain "def" (not async): the model call can take a while, so FastAPI runs it in a worker thread.
@app.post("/ask")
def ask(req: AskRequest):
    return answer_question(req.dataset, req.question)
