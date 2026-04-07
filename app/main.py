import os
import logging
import shutil

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.pdf_loader import load_and_split_pdf
from app.rag_pipeline import (
    get_embeddings,
    load_vectorstore,
    create_vectorstore,
    add_documents,
    create_qa_chain,
    ask_question,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

app = FastAPI(title="RAG Assistant API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vectorstore = None
rag_chain   = None
retriever   = None


class QueryRequest(BaseModel):
    query: str


@app.on_event("startup")
def startup():
    global vectorstore, rag_chain, retriever
    log.info("Startup — checking for existing vectorstore...")
    try:
        embeddings  = get_embeddings()
        vectorstore = load_vectorstore(embeddings)
        if vectorstore is not None:
            rag_chain, retriever = create_qa_chain(vectorstore)
            log.info("✅ Vectorstore + chain loaded from disk.")
        else:
            log.info("ℹ️  No vectorstore found. Upload a PDF to begin.")
    except Exception as e:
        log.error(f"❌ Startup error: {e}")


@app.get("/")
def root():
    return {
        "status":            "running",
        "vectorstore_ready": vectorstore is not None,
        "chain_ready":       rag_chain is not None,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset():
    """Wipe vectorstore from disk and memory for a clean start."""
    global vectorstore, rag_chain, retriever
    VECTORSTORE_PATH = "vectorstore/faiss_index"
    try:
        if os.path.exists(VECTORSTORE_PATH):
            shutil.rmtree(VECTORSTORE_PATH)
            log.info("🗑️  Vectorstore deleted.")
        vectorstore = None
        rag_chain   = None
        retriever   = None
        return {"message": "✅ Reset complete. Upload a new PDF to start fresh."}
    except Exception as e:
        log.error(f"❌ Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global vectorstore, rag_chain, retriever

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    log.info(f"📄 Received: {file.filename}")

    try:
        os.makedirs("data", exist_ok=True)
        path = f"data/{file.filename}"

        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        with open(path, "wb") as f:
            f.write(contents)
        log.info(f"💾 Saved: {path} ({len(contents):,} bytes)")

        docs = load_and_split_pdf(path)
        if not docs:
            raise HTTPException(
                status_code=422,
                detail="No text could be extracted. Is this a scanned PDF?"
            )
        log.info(f"✂️  {len(docs)} chunks created")

        embeddings = get_embeddings()

        # Always create a FRESH vectorstore — no stale data mixing in
        vectorstore = create_vectorstore(docs, embeddings)
        log.info("🗄️  Fresh vectorstore created and saved.")

        rag_chain, retriever = create_qa_chain(vectorstore)
        log.info("⛓️  RAG chain ready.")

        return {
            "message": f"✅ '{file.filename}' uploaded and indexed successfully.",
            "chunks":  len(docs),
            "status":  "ready",
        }

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"❌ Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/ask")
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
        log.info(f"✅ Answer ready ({len(answer)} chars, {len(docs)} source chunks)")

        sources = []
        for d in docs:
            text = d.page_content.strip()
            sources.append(text[:400] + ("…" if len(text) > 400 else ""))

        return {"answer": answer, "sources": sources}

    except Exception as e:
        log.error(f"❌ Ask error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")