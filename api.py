"""
api.py
FastAPI REST endpoint for PixelRAG.

Endpoints:
  GET  /health          — liveness check
  POST /ingest          — upload + index a PDF or image file
  GET  /index           — list indexed files
  DELETE /index         — clear the entire index
  POST /search          — retrieve relevant pages for a query (no answer)
  POST /ask             — full pipeline: retrieve + rerank + synthesise answer
"""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

load_dotenv()

from embed_client import EmbedClient, EmbedServerError
from ingest import ingest_file, clear_index, list_indexed_files
from search import search
from answer import synthesise_answer

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PixelRAG API",
    description="Visual document retrieval and question answering — no text parsing.",
    version="1.0.0",
)

# ── Shared embed client ───────────────────────────────────────────────────────

def _get_client() -> EmbedClient:
    url = os.environ.get("EMBED_API_URL", "")
    if not url:
        raise HTTPException(status_code=503, detail="EMBED_API_URL is not set in .env")
    return EmbedClient(base_url=url)


# ── Request / Response models ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    query: str
    top_k: int = 5

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class PageResult(BaseModel):
    source: str
    page: int
    score: float

class SearchResponse(BaseModel):
    query: str
    results: list[PageResult]

class AskResponse(BaseModel):
    query: str
    answer: str
    retrieved: list[PageResult]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", summary="Liveness check")
def health():
    """Returns OK if the API is running. Also checks the embed server."""
    client = _get_client()
    try:
        embed_status = client.health()
    except Exception as e:
        embed_status = {"error": str(e)}
    return {"status": "ok", "embed_server": embed_status}


@app.post("/ingest", summary="Upload and index a PDF or image file")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF or image file. The file is rendered to page images,
    embedded via the Colab server, and added to the FAISS index.
    """
    suffix = Path(file.filename).suffix.lower()
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    client = _get_client()

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    tmp_path.write_bytes(await file.read())

    logs = []
    try:
        count = ingest_file(tmp_path, client, progress_cb=logs.append)
    except EmbedServerError as e:
        raise HTTPException(status_code=502, detail=f"Embed server error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return {"file": file.filename, "pages_indexed": count, "log": logs}


@app.get("/index", summary="List all indexed files")
def index_list():
    """Returns the list of files currently in the FAISS index."""
    return {"indexed_files": list_indexed_files()}


@app.delete("/index", summary="Clear the entire index")
def index_clear():
    """Deletes the FAISS index, metadata, and all tile images."""
    clear_index()
    return {"status": "index cleared"}


@app.post("/search", response_model=SearchResponse, summary="Retrieve relevant pages")
def search_endpoint(req: SearchRequest):
    """
    Embed the query and retrieve the top-k most relevant pages using
    FAISS + MMR. Returns page metadata and similarity scores — no answer synthesis.
    """
    client = _get_client()
    try:
        results = search(req.query, client, top_k=req.top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except EmbedServerError as e:
        raise HTTPException(status_code=502, detail=f"Embed server error: {e}")

    return SearchResponse(
        query=req.query,
        results=[
            PageResult(source=r.source, page=r.page, score=round(float(r.score), 4))
            for r in results
        ],
    )


@app.post("/ask", response_model=AskResponse, summary="Retrieve + synthesise answer")
def ask_endpoint(req: AskRequest):
    """
    Full pipeline: embed query → FAISS + MMR retrieval → rerank → crop →
    gpt-4o answer synthesis. Returns the answer and the retrieved pages.
    """
    client = _get_client()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set in .env")

    try:
        results = search(req.query, client, top_k=req.top_k)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except EmbedServerError as e:
        raise HTTPException(status_code=502, detail=f"Embed server error: {e}")

    if not results:
        raise HTTPException(status_code=404, detail="No relevant pages found in the index.")

    try:
        answer = synthesise_answer(req.query, results, api_key=api_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer synthesis failed: {e}")

    return AskResponse(
        query=req.query,
        answer=answer,
        retrieved=[
            PageResult(source=r.source, page=r.page, score=round(float(r.score), 4))
            for r in results
        ],
    )
