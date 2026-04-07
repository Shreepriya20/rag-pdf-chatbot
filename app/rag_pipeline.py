import os
import shutil

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

VECTORSTORE_PATH = "vectorstore/faiss_index"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_vectorstore(embeddings):
    if os.path.exists(VECTORSTORE_PATH):
        return FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return None


def create_vectorstore(documents, embeddings):
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)
    os.makedirs("vectorstore", exist_ok=True)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore


def add_documents(vectorstore, documents):
    new_db = FAISS.from_documents(documents, vectorstore.embedding_function)
    vectorstore.merge_from(new_db)
    vectorstore.save_local(VECTORSTORE_PATH)
    return vectorstore


def create_qa_chain(vector_store):
    llm = OllamaLLM(
        model="mistral",
        temperature=0.1,
        num_predict=300,      # ← SPEED FIX: cap output tokens
        num_ctx=2048,         # ← SPEED FIX: smaller context window
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # Short, tight prompt = faster inference
    prompt = ChatPromptTemplate.from_template(
        "Answer using ONLY the context below. Be concise.\n"
        "If not found, say: 'Not found in the document.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context":  retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


def ask_question(rag_chain, retriever, query):
    answer = rag_chain.invoke(query)
    docs   = retriever.invoke(query)
    return answer, docs