from dotenv import load_dotenv
load_dotenv()

import os
import uuid
import threading
import traceback
from typing import List, Optional, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from chromadb import Client
from chromadb.config import Settings
import shutil
import asyncio
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import math
import time
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")  # 🔥 usar service role!

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise Exception("Credenciais do Supabase não encontradas")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

BUCKET_NAME = "sped-documents"


app = FastAPI()

from fastapi.responses import Response

from fastapi import Request

from fastapi.responses import JSONResponse

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")

if not INTERNAL_API_KEY:
    raise Exception("INTERNAL_API_KEY não configurada")

PUBLIC_ROUTES = ["/status", "/docs", "/openapi.json"]

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    if any(request.url.path.startswith(route) for route in PUBLIC_ROUTES):
        return await call_next(request)

    api_key = request.headers.get("x-api-key")

    if api_key != INTERNAL_API_KEY:
        return JSONResponse(
            status_code=401,
            content={"detail": "Unauthorized"}
        )

    return await call_next(request)


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
PERSIST_DIR = "/data/chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY não encontrada")

# -------------------------------
# EMBEDDINGS
# -------------------------------
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# -------------------------------
# VECTOR STORE
# -------------------------------
def get_vector_store(project_id: str):
    try:
        project_path = os.path.join(PERSIST_DIR, project_id)

        return Chroma(
            collection_name="default",  # 🔥 sempre fixo agora
            persist_directory=project_path,  # 🔥 pasta por projeto
            embedding_function=embeddings
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro vector store: {str(e)}")
    


    # ===============================
# MÉTRICAS (ENRICHMENT)
# ===============================
def generate_basic_metrics(context: str):
    lines = context.split("\n")

    metrics = {
        "registros_c100": 0,
        "registros_c170": 0,
        "possiveis_divergencias": 0
    }

    for line in lines:
        if "|C100|" in line:
            metrics["registros_c100"] += 1
        if "|C170|" in line:
            metrics["registros_c170"] += 1

    if metrics["registros_c170"] > metrics["registros_c100"] * 5:
        metrics["possiveis_divergencias"] += 1

    return metrics

# -------------------------------
# SPLITTER
# -------------------------------
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
# JOBS
# -------------------------------
jobs = {}

# ===============================
# RAG MELHORADO
# ===============================
def get_context(query: str, project_id: str, k: int = 10):
    vector_store = get_vector_store(project_id)

    queries = [
        query,
        "divergência C100 C170",
        "créditos PIS COFINS",
        "ICMS base cálculo",
        "NCM CST erro"
    ]

    docs = []
    for q in queries:
        docs.extend(vector_store.similarity_search(q, k=k))

    # remove duplicados
    seen = set()
    unique_docs = []
    for d in docs:
        key = d.page_content
        if key not in seen:
            seen.add(key)
            unique_docs.append(d)

    if not unique_docs:
        return "Sem dados"

    return "\n\n".join([
        f"[{d.metadata.get('source')}] {d.page_content}"
        for d in unique_docs[:20]
    ])

# ===============================
# PROMPT EXECUTIVO
# ===============================
def build_prompt(context: str, enrichment: dict):

    return f"""
Você é um CONSULTOR ESTRATÉGICO TRIBUTÁRIO especializado em SPED.

Gere um RELATÓRIO EXECUTIVO para CEO/CFO.

====================
CONTEXTO
====================
{context}

====================
DADOS ANALÍTICOS
====================
{enrichment}

====================
INSTRUÇÕES
====================

O relatório deve conter:

1. SUMÁRIO EXECUTIVO
2. PRINCIPAIS RISCOS FISCAIS (quantificar)
3. ACHADOS CONSOLIDADOS
4. OPORTUNIDADES ESTRATÉGICAS
5. RECOMENDAÇÕES (curto e médio prazo)
6. IMPACTO FINANCEIRO (estimado)
7. CONCLUSÃO

REGRAS:
- Sempre quantificar (ex: 10–30 notas)
- Sempre estimar impacto financeiro
- Classificar risco: Alto / Médio / Baixo
- Não ser genérico

FORMATO:
Texto estruturado (estilo consultoria Big4)
"""

# ===============================
# REQUEST
# ===============================
class SummaryRequest(BaseModel):
    project_id: str
    query: Optional[str] = "análise geral"
# -------------------------------
# WORKER
# -------------------------------
def process_job(job_id: str, files_data: List[dict], project_id: str):
    try:
        job = jobs[job_id]
        job["status"] = "processing"

        vector_store = get_vector_store(project_id)
        all_chunks = []
        all_metadata = []

        # --- ETAPA 1: CHUNKING ---
        job["stage"] = "chunking"
        local_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        with ProcessPoolExecutor() as executor:
            texts = [f["text"] for f in files_data]
            results = list(executor.map(local_splitter.split_text, texts))

        for i, chunks in enumerate(results):
            filename = files_data[i]["filename"]
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": filename,
                    "chunk_index": idx,
                    "project_id": project_id
                })
        
        job["progress"] = 50

        # --- ETAPA 2: EMBEDDING COM RETRY E ESPAÇAMENTO ---
        job["stage"] = "embedding"
        
        BATCH_SIZE = 100 
        num_chunks = len(all_chunks)
        num_batches = (num_chunks + BATCH_SIZE - 1) // BATCH_SIZE

        def add_batch(batch_idx):
            # Adicionamos uma lógica de tentativa (retry) para caso a API falhe
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    start = batch_idx * BATCH_SIZE
                    end = min(start + BATCH_SIZE, num_chunks)
                    vector_store.add_texts(
                        texts=all_chunks[start:end],
                        metadatas=all_metadata[start:end]
                    )
                    return batch_idx
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1)) # Espera um pouco mais a cada erro
                        continue
                    raise e

        # Usamos apenas 1 worker para garantir ordem e evitar picos de TPM
        # No seu Tier atual, a velocidade de 1 worker sequencial com lote de 100
        # já será muito rápida e segura.
        with ThreadPoolExecutor(max_workers=1) as t_executor:
            futures = []
            for i in range(num_batches):
                futures.append(t_executor.submit(add_batch, i))
                # --- O SEGREDO AQUI ---
                # Pequena pausa de 0.2s entre o disparo de cada lote para suavizar o TPM
                time.sleep(0.2) 
            
            for idx, future in enumerate(futures):
                future.result() 
                job["progress"] = 50 + int(((idx + 1) / num_batches) * 50)

        job["status"] = "completed"
        job["stage"] = "done"
        job["progress"] = 100

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        print(f"Erro Crítico no Job {job_id}:")
        print(traceback.format_exc())

# -------------------------------
# UPLOAD
# -------------------------------
@app.post("/upload")
async def upload_documents(project_id: str, files: List[UploadFile] = File(...)):
    try:
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id obrigatório")

        job_id = str(uuid.uuid4())

        # Função auxiliar para processar cada arquivo individualmente de forma assíncrona
        async def process_single_file(file: UploadFile):
            # Leitura assíncrona do conteúdo
            content = await file.read()
            # Decodificação (aqui você já ganha tempo processando enquanto outros leem)
            text = content.decode("utf-8", errors="ignore")
            
            if text.strip():
                return {
                    "filename": file.filename,
                    "text": text
                }
            return None

        # --- O PONTO CHAVE: Upload Paralelo e Assíncrono ---
        # asyncio.gather dispara todas as corrotinas ao mesmo tempo
        # Em vez de esperar arquivo por arquivo, o Python gerencia o I/O de todos simultaneamente
        results = await asyncio.gather(*[process_single_file(f) for f in files])

        # Filtra arquivos que retornaram None (vazios)
        files_data = [res for res in results if res is not None]

        if not files_data:
            raise HTTPException(status_code=400, detail="Nenhum arquivo válido")

        # Registro do Job
        jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "stage": "upload",
            "project_id": project_id
        }

        # Inicia o processamento pesado (Chunking/Embedding) em uma thread separada
        # para não bloquear a resposta do endpoint
        threading.Thread(
            target=process_job,
            args=(job_id, files_data, project_id)
        ).start()

        return {"job_id": job_id, "project_id": project_id}

    except Exception as e:
        # Se algo falhar no gather ou no processamento inicial
        raise HTTPException(status_code=500, detail=str(e))
# -------------------------------
# STATUS
# -------------------------------



@app.delete("/delete-project/{project_id}")
def delete_project(project_id: str, folder_path: str):
    try:
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id obrigatório")

        if not folder_path:
            raise HTTPException(status_code=400, detail="folder_path obrigatório")

        # -------------------------------
        # 🔥 1. DELETAR STORAGE (LOOP)
        # -------------------------------
        try:
            while True:
                files = supabase.storage.from_(BUCKET_NAME).list(path=folder_path)

                if not files:
                    break  # acabou tudo

                paths = [
                    f"{folder_path.rstrip('/')}/{file['name']}"
                    for file in files
                ]

                supabase.storage.from_(BUCKET_NAME).remove(paths)

                print(f"🗑️ Deletados {len(paths)} arquivos...")

                # 🔥 pequena pausa pra evitar rate limit
                time.sleep(0.2)

        except Exception as e:
            print("⚠️ Erro ao deletar storage:", str(e))

        # -------------------------------
        # 🔥 2. DELETAR CHROMA
        # -------------------------------
        project_path = os.path.join(PERSIST_DIR, project_id)

        if os.path.exists(project_path):
            shutil.rmtree(project_path)

        return {
            "status": "success",
            "message": f"Project {project_id} + folder {folder_path} deletados completamente"
        }

    except Exception as e:
        print("🔥 ERRO AO DELETAR PROJECT")
        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=f"Erro ao deletar projeto: {str(e)}"
        )


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    return job

# -------------------------------
# SUMMARY (COM LOGS 🔥)
# -------------------------------
@app.post("/generate-summary")
async def generate_summary(req: SummaryRequest):
    try:
        context = get_context(req.query, req.project_id, 5)

        metrics = generate_basic_metrics(context)

        prompt = build_prompt(context, metrics)

        response = llm.invoke(prompt)

        return {
            "summary": response.content
        }

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(500, str(e))