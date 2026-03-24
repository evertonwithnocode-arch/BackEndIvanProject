from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import threading
from typing import List, Optional, Dict

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

app = FastAPI()

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# CONFIG
# -------------------------------
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
PERSIST_DIR = "./chroma_db"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY não encontrada")

# -------------------------------
# EMBEDDINGS + VECTOR STORE
# -------------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

vector_store = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embeddings
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# -------------------------------
# LLM
# -------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=OPENAI_API_KEY
)

# -------------------------------
# JOB STORAGE
# -------------------------------
jobs = {}

# -------------------------------
# RAG: RECUPERAR CONTEXTO
# -------------------------------
def get_context(query: str, k: int = 10):
    docs = vector_store.similarity_search(query, k=k)

    context = "\n\n".join([
        f"[Fonte: {doc.metadata.get('source')} | Chunk: {doc.metadata.get('chunk_index')}]\n{doc.page_content}"
        for doc in docs
    ])

    return context

# -------------------------------
# PROMPT BUILDER
# -------------------------------
def build_prompt(template: str, context: str, enrichment: dict | None):

    enrichment_text = ""
    if enrichment:
        enrichment_text = f"""
====================
DADOS DE ENRIQUECIMENTO
====================
{enrichment}
"""

    prompt = f"""
Você é um especialista em análise de documentos SPED.

Gere um sumário analítico COMPLETO.

====================
CONTEXTO
====================
{context}

{enrichment_text}

====================
INSTRUÇÕES DO USUÁRIO
====================
{template}

====================
REGRAS
====================
- Gere insights relevantes
- Faça cruzamento de dados
- Identifique inconsistências
- Sugira oportunidades fiscais
- Inclua justificativas
- Cite as fontes (arquivo + trecho)
- Estruture a resposta

====================
FORMATO DE SAÍDA
====================
Responda em JSON com:

- insights
- inconsistencias
- oportunidades
- analises
- referencias
"""

    return prompt

# -------------------------------
# REQUEST MODEL
# -------------------------------
class SummaryRequest(BaseModel):
    template: str
    query: Optional[str] = "gerar sumário geral"
    enrichment: Optional[Dict] = None
    k: Optional[int] = 10

# -------------------------------
# WORKER
# -------------------------------
def process_job(job_id: str, files_data: List[dict]):

    job = jobs[job_id]
    job["status"] = "processing"

    total_files = len(files_data)
    job["total_files"] = total_files

    all_chunks = []
    all_metadata = []

    job["stage"] = "chunking"

    for i, file in enumerate(files_data):
        text = file["text"]
        filename = file["filename"]

        chunks = text_splitter.split_text(text)

        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": filename,
                "chunk_index": idx
            })

        job["progress"] = int((i + 1) / total_files * 50)

    job["stage"] = "embedding"

    BATCH_SIZE = 100
    total_chunks = len(all_chunks)

    for i in range(0, total_chunks, BATCH_SIZE):
        batch_chunks = all_chunks[i:i + BATCH_SIZE]
        batch_meta = all_metadata[i:i + BATCH_SIZE]

        vector_store.add_texts(
            texts=batch_chunks,
            metadatas=batch_meta
        )

        job["progress"] = 50 + int((i + BATCH_SIZE) / total_chunks * 50)

    job["status"] = "completed"
    job["stage"] = "done"
    job["progress"] = 100

# -------------------------------
# UPLOAD
# -------------------------------
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):

    job_id = str(uuid.uuid4())
    files_data = []

    for file in files:
        content = await file.read()
        text = content.decode("utf-8", errors="ignore")

        if not text.strip():
            continue

        files_data.append({
            "filename": file.filename,
            "text": text
        })

    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "stage": "upload",
        "total_files": len(files_data),
        "processed_files": 0
    }

    thread = threading.Thread(target=process_job, args=(job_id, files_data))
    thread.start()

    return {
        "success": True,
        "job_id": job_id
    }

# -------------------------------
# STATUS
# -------------------------------
@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)

    if not job:
        return {"error": "Job não encontrado"}

    return job

# -------------------------------
# SEARCH
# -------------------------------
@app.post("/search")
async def search(query: str, k: int = 5):

    results = vector_store.similarity_search(query, k=k)

    return {
        "query": query,
        "results": [
            {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in results
        ]
    }

# -------------------------------
# GENERATE SUMMARY (RAG + IA)
# -------------------------------
@app.post("/generate-summary")
async def generate_summary(req: SummaryRequest):

    # 1. RAG
    context = get_context(req.query, req.k)

    # 2. Prompt
    prompt = build_prompt(
        template=req.template,
        context=context,
        enrichment=req.enrichment
    )

    # 3. LLM
    response = llm.invoke(prompt)

    return {
        "summary": response.content,
        "query": req.query,
        "chunks_used": req.k
    }