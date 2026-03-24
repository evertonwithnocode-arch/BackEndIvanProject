from dotenv import load_dotenv
load_dotenv()

import os
import tempfile
import uuid
import threading
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
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
# JOB STORAGE (fila simples em memória)
# -------------------------------
jobs = {}  
# Estrutura:
# job_id: {
#   status: "pending | processing | completed",
#   progress: 0-100,
#   stage: "upload | chunking | embedding | done",
#   total_files: int,
#   processed_files: int
# }

# -------------------------------
# FUNÇÃO WORKER (RODA EM BACKGROUND)
# -------------------------------
def process_job(job_id: str, files_data: List[dict]):
    """
    Essa função roda em background (thread)
    Responsável por:
    1. Chunking
    2. Embeddings
    3. Atualizar progresso
    """

    job = jobs[job_id]
    job["status"] = "processing"

    total_files = len(files_data)
    job["total_files"] = total_files

    all_chunks = []
    all_metadata = []

    # -------------------------------
    # ETAPA 1: CHUNKING
    # -------------------------------
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

        # progresso parcial (0 → 50%)
        job["progress"] = int((i + 1) / total_files * 50)

    # -------------------------------
    # ETAPA 2: EMBEDDINGS + STORAGE
    # -------------------------------
    job["stage"] = "embedding"

    BATCH_SIZE = 100

    total_chunks = len(all_chunks)

    for i in range(0, total_chunks, BATCH_SIZE):
        batch_chunks = all_chunks[i:i + BATCH_SIZE]
        batch_meta = all_metadata[i:i + BATCH_SIZE]

        # Chroma já gera embeddings automaticamente
        vector_store.add_texts(
            texts=batch_chunks,
            metadatas=batch_meta
        )

        # progresso (50 → 100%)
        job["progress"] = 50 + int((i + BATCH_SIZE) / total_chunks * 50)

    # -------------------------------
    # FINALIZAÇÃO
    # -------------------------------
    job["status"] = "completed"
    job["stage"] = "done"
    job["progress"] = 100


# -------------------------------
# UPLOAD (CRIA JOB)
# -------------------------------
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Endpoint rápido (NÃO processa tudo aqui)
    
    Faz:
    1. Recebe arquivos
    2. Cria job_id
    3. Dispara worker em background
    4. Retorna job_id para o frontend
    
    Frontend deve usar /status/{job_id}
    """

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

    # cria job inicial
    jobs[job_id] = {
        "status": "pending",
        "progress": 0,
        "stage": "upload",
        "total_files": len(files_data),
        "processed_files": 0
    }

    # dispara worker em background
    thread = threading.Thread(target=process_job, args=(job_id, files_data))
    thread.start()

    return {
        "success": True,
        "job_id": job_id,
        "message": "Processamento iniciado"
    }


# -------------------------------
# STATUS (POLLING)
# -------------------------------
@app.get("/status/{job_id}")
def get_status(job_id: str):
    """
    Frontend chama isso a cada X segundos
    
    Retorna:
    - status
    - progresso
    - etapa atual
    """

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