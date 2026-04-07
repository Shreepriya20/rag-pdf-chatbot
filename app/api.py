import os
import shutil
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.rag_pipeline import (
    get_embeddings,
    load_vectorstore,
    create_vectorstore,
    create_qa_chain,
    ask_question,
)
from app.pdf_loader import load_and_split_pdf

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="RAG Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────
vectorstore = None
rag_chain   = None
retriever   = None
embeddings  = get_embeddings()   # load once at startup

# ── Request model ─────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str

# ── Routes ────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message":           "RAG API Running 🚀",
        "vectorstore_ready": vectorstore is not None,
        "chain_ready":       rag_chain is not None,
    }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore, rag_chain, retriever

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", file.filename)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        log.info(f"💾 Saved: {file_path}")

        # Load + split
        documents = load_and_split_pdf(file_path)
        if not documents:
            raise HTTPException(status_code=422, detail="No text extracted from PDF.")
        log.info(f"✂️  {len(documents)} chunks created")

        # Always create FRESH vectorstore — wipes old stale index
        vectorstore = create_vectorstore(documents, embeddings)
        log.info("🗄️  Fresh vectorstore created.")

        # Build chain
        rag_chain, retriever = create_qa_chain(vectorstore)
        log.info("⛓️  RAG chain ready.")

        return {
            "message": f"✅ '{file.filename}' uploaded and indexed successfully.",
            "chunks":  len(documents),
            "status":  "ready",
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/ask")             # POST — matches ui.py fetch call
async def ask(request: QueryRequest):
    global rag_chain, retriever

    question = request.query.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    log.info(f"❓ {question}")

    if rag_chain is None or retriever is None:
        return {
            "answer":  "⚠️ No document loaded. Please upload a PDF first.",
            "sources": [],
        }

    try:
        answer, docs = ask_question(rag_chain, retriever, question)
        log.info(f"✅ {len(answer)} chars | {len(docs)} source chunks")

        sources = [
            d.page_content.strip()[:400] + ("…" if len(d.page_content) > 400 else "")
            for d in docs
        ]

        return {"answer": answer, "sources": sources}

    except Exception as e:
        log.error(f"❌ Ask error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")