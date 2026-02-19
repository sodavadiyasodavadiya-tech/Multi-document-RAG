import os
from typing import List
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from vectordb import VectorStore
from rag_pipeline import build_index, answer_multiple_questions

app = FastAPI()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# embedding dimension for BGE-Small = 384
vectordb = VectorStore(dim=384)


class QuestionRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a single document"""
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = build_index(file_path, vectordb)
    return {"status": "success", "result": result}


@app.post("/upload-multiple")
async def upload_multiple_documents(files: List[UploadFile] = File(...)):
    """
    Upload and index multiple documents sequentially (no parallel processing).
    Processes files one by one in the order they were uploaded.
    """
    results = []
    successful = 0
    failed = 0
    
    for file in files:
        try:
            # Save file
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            
            # Index file
            index_result = build_index(file_path, vectordb)
            
            results.append({
                "filename": file.filename,
                "status": "success",
                "chunks_added": index_result.get("chunks_added", 0),
                "source": index_result.get("source", file.filename)
            })
            successful += 1
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
            failed += 1
    
    return {
        "status": "success",
        "total_files": len(files),
        "successful": successful,
        "failed": failed,
        "results": results
    }


@app.post("/ask")
async def ask_question(req: QuestionRequest):
    """Ask a question and get answers from indexed documents"""
    result = answer_multiple_questions(req.question, vectordb)
    return {"status": "success", "result": result}

