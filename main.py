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
import json

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
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

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
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)


def run_multi_step_rag(query, project_id, template):
    vector_store = get_vector_store(project_id)

    docs = vector_store.max_marginal_relevance_search(
        query, k=50, fetch_k=100
    )  # 🔥 aumenta MUITO

    partial_results = []

    for doc in docs:
        prompt = f"""
        Extraia deste trecho:
        - inconsistências
        - valores financeiros
        - padrões fiscais
        - evidências
        Se não houver dado concreto, responda: "SEM EVIDÊNCIA"
        Texto:
        DOCUMENTO: {doc.metadata.get("source")}
        CHUNK: {doc.metadata.get("chunk_index")}

        TEXTO:
        {doc.page_content}
        """

        res = llm.invoke(prompt)
        partial_results.append(res.content)

    aggregated = "\n\n".join(partial_results)

    final_prompt = f"""
    {template}
    REGRAS CRÍTICAS:

    - PROIBIDO inventar valores financeiros
    - PROIBIDO estimativas sem cálculo explícito
    - PROIBIDO linguagem genérica

  - Cada insight deve conter:
    - trecho real
    - origem (arquivo + chunk)
    - cálculo demonstrado
    BASE DE DADOS CONSOLIDADA:
    {aggregated}
    """

    final = llm.invoke(final_prompt)

    return final.content


# -------------------------------
# VECTOR STORE
# -------------------------------
def get_vector_store(project_id: str):
    try:
        project_path = os.path.join(PERSIST_DIR, project_id)

        return Chroma(
            collection_name=f"project_{project_id}",  # 🔥 sempre fixo agora
            persist_directory=project_path,  # 🔥 pasta por projeto
            embedding_function=embeddings,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro vector store: {str(e)}")


# -------------------------------
# SPLITTER
# -------------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)

# -------------------------------
# LLM
# -------------------------------
llm = ChatOpenAI(model="gpt-4.1", temperature=0.0, api_key=OPENAI_API_KEY)

# -------------------------------
# JOBS
# -------------------------------
jobs = {}
summary_jobs = {}


# -------------------------------
# RAG
# -------------------------------
def get_context(query: str, project_id: str, k: int = 10):
    try:
        vector_store = get_vector_store(project_id)

        docs = vector_store.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=k * 4
        )

        if not docs:
            return "Nenhum dado encontrado para este projeto."

        context_parts = []

        for doc in docs:

            doc_type = doc.metadata.get("type", "document")

            if doc_type == "enrichment":
                prefix = "[ENRICHMENT]"
            else:
                prefix = "[DOCUMENTO]"

            context_parts.append(
                f"""
{prefix}
Fonte: {doc.metadata.get("source")}
Chunk: {doc.metadata.get("chunk_index")}

{doc.page_content}
"""
            )

        return "\n\n".join(context_parts)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao buscar contexto: {str(e)}"
        )

# -------------------------------
# PROMPT
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
Você é um auditor fiscal especialista em SPED (EFD PIS/COFINS).

Sua função é identificar inconsistências reais, validar cálculos e garantir integridade dos dados.

====================
CONTEXTO
====================
{context}

{enrichment_text}

====================
INSTRUÇÕES
====================
{template}

====================
REGRAS DE AUDITORIA (OBRIGATÓRIAS)
====================

1. NÃO gere descrições genéricas.
   ❌ Proibido: "volume significativo", "análise necessária"
   ✅ Obrigatório: achados concretos

2. Sempre validar consistência entre registros:
   - A100 (documento) vs A170 (itens)
   - Totais vs soma dos itens
   - Valores repetidos ou divergentes

3. Verificar:
   - Base de cálculo
   - Alíquota aplicada
   - Valor do tributo
   - Coerência entre eles

4. Identificar possíveis erros como:
   - Divergência entre total e itens
   - Valores duplicados
   - CST incompatível
   - Campos zerados indevidamente
   - Dados inconsistentes entre arquivos

5. Quando NÃO houver erro:
   - Dizer explicitamente: "Nenhuma inconsistência relevante encontrada"

6. Toda análise deve conter:
   - Evidência (trecho real)
   - Explicação técnica
   - Lógica aplicada

7. Se fizer cálculo:
   - Mostrar fórmula
   - Mostrar valores usados
   - Mostrar resultado

====================
Se o usuário pedir estrutura → respeite
Senão → liberdade total
====================

{{
  "insights": [
    {{
      "titulo": "",
      "explicacao": "",
      "passo_a_passo": [],
      "dados_utilizados": [],
      "logica_aplicada": "",
      "conclusao": ""
    }}
  ],
  "inconsistencias": [
    {{
      "titulo": "",
      "descricao": "",
      "impacto": "",
      "evidencias": [
        {{
          "fonte": "",
          "trecho": ""
        }}
      ],
      "recomendacao": ""
    }}
  ],
  "analises": [],
  "referencias": []
}}

====================
OBJETIVO FINAL
====================

Gerar um relatório técnico de auditoria, focado em:
- detectar problemas reais
- validar integridade dos dados
- permitir verificação por auditor humano

Se não houver inconsistências, deixe isso claro.
"""

    return prompt


# -------------------------------
# REQUEST
# -------------------------------
class SummaryRequest(BaseModel):
    template: str
    query: Optional[str] = "gerar sumário geral"
    enrichment: Optional[Dict] = None
    k: Optional[int] = 20
    project_id: str


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
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        with ProcessPoolExecutor() as executor:
            texts = [f["text"] for f in files_data]
            results = list(executor.map(local_splitter.split_text, texts))

        for i, chunks in enumerate(results):
            filename = files_data[i]["filename"]
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append(
                   {
                       "source": filename,
                       "chunk_index": idx,
                       "project_id": project_id,
                       "type": "document"
                   }
               )

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
                        texts=all_chunks[start:end], metadatas=all_metadata[start:end]
                    )
                    return batch_idx
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        time.sleep(
                            2 * (attempt + 1)
                        )  # Espera um pouco mais a cada erro
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


def enrichment_to_text(data, parent_key=""):
    texts = []

    if isinstance(data, dict):
        for key, value in data.items():
            new_key = f"{parent_key}.{key}" if parent_key else key
            texts.append(enrichment_to_text(value, new_key))

    elif isinstance(data, list):
        for idx, item in enumerate(data):
            texts.append(enrichment_to_text(item, f"{parent_key}[{idx}]"))

    else:
        texts.append(f"{parent_key}: {data}")

    return "\n".join(texts)

class EnrichmentRequest(BaseModel):
    project_id: str
    enrichment: Dict
    source: Optional[str] = "manual_enrichment"

@app.post("/enrichment")
async def upload_enrichment(req: EnrichmentRequest):
    try:
        vector_store = get_vector_store(req.project_id)

        # -------------------------
        # SERIALIZA JSON
        # -------------------------
        enrichment_text = enrichment_to_text(req.enrichment)

        # -------------------------
        # CHUNKING
        # -------------------------
        chunks = text_splitter.split_text(enrichment_text)

        texts = []
        metadatas = []

        for idx, chunk in enumerate(chunks):
            texts.append(chunk)

            metadatas.append({
                "project_id": req.project_id,
                "type": "enrichment",
                "source": req.source,
                "chunk_index": idx
            })

        # -------------------------
        # EMBEDDING + SAVE
        # -------------------------
        vector_store.add_texts(
            texts=texts,
            metadatas=metadatas
        )

        return {
            "status": "success",
            "chunks_saved": len(chunks)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
                return {"filename": file.filename, "text": text}
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
            "project_id": project_id,
        }

        # Inicia o processamento pesado (Chunking/Embedding) em uma thread separada
        # para não bloquear a resposta do endpoint
        threading.Thread(
            target=process_job, args=(job_id, files_data, project_id)
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

                paths = [f"{folder_path.rstrip('/')}/{file['name']}" for file in files]

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
            "message": f"Project {project_id} + folder {folder_path} deletados completamente",
        }

    except Exception as e:
        print("🔥 ERRO AO DELETAR PROJECT")
        print(traceback.format_exc())

        raise HTTPException(
            status_code=500, detail=f"Erro ao deletar projeto: {str(e)}"
        )


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")

    return job


@app.get("/summary-status/{job_id}")
def get_summary_status(job_id: str):
    job = summary_jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Summary job não encontrado")

    return job


def process_summary_job(job_id: str, req: SummaryRequest):
    try:
        job = summary_jobs[job_id]

        job["status"] = "processing"
        job["stage"] = "starting"
        print(f"[SUMMARY][{job_id}] 🚀 START")

        # -------------------------------
        # 🔥 DETECT MODE
        # -------------------------------
        if "DOCUMENTO 1" in req.template:
            mode = "strategic"
        else:
            mode = "audit"

        job["mode"] = mode
        print(f"[SUMMARY][{job_id}] Mode: {mode}")

        # ===============================
        # 🔵 AUDIT MODE
        # ===============================
        if mode == "audit":
            job["stage"] = "retrieving_context"

            context = get_context(req.query, req.project_id, req.k)

            print(f"[SUMMARY][{job_id}] Context size: {len(context)}")

            context = context[:12000]

            job["stage"] = "building_prompt"
            prompt = build_prompt(req.template, context, req.enrichment)

            print(f"[SUMMARY][{job_id}] Prompt size: {len(prompt)}")

            job["stage"] = "llm_call"
            response = llm.invoke(prompt)

            job["result"] = response.content

        # ===============================
        # 🔴 STRATEGIC MODE
        # ===============================
        else:
            job["stage"] = "multi_step_rag"

            vector_store = get_vector_store(req.project_id)

            # 🔥 REDUZIDO (menos ruído e menos hallucination)
            docs = vector_store.max_marginal_relevance_search(
                req.query, k=20, fetch_k=40
            )

            print(f"[SUMMARY][{job_id}] Docs encontrados: {len(docs)}")

            partial_results = []

            # -------------------------------
            # 🔵 EXTRAÇÃO ESTRUTURADA (CRÍTICO)
            # -------------------------------
            for i, doc in enumerate(docs):
                print(f"[SUMMARY][{job_id}] Doc {i+1}/{len(docs)}")

                prompt = f"""
                Extraia APENAS dados REAIS do texto.

                RETORNE JSON:

                {{
                "evidencias": [
                    {{
                      "documento": "{doc.metadata.get("source")}",
                      "chunk": "{doc.metadata.get("chunk_index")}",
                      "registro": "",
                      "trecho": "",
                      "tipo": "financeiro | inconsistencia | fiscal"
                    }}
                  ]
                }}

                REGRAS:
                - NÃO inventar nada
                - NÃO resumir
                - NÃO estimar valores
                - Se não houver evidência → retornar lista vazia

                TEXTO:
                {doc.page_content}
                """

                try:
                    res = llm.invoke(prompt)
                    content = res.content.strip()

                    # 🔥 filtro mínimo
                    if content and "evidencias" in content:
                        partial_results.append(content)

                except Exception as e:
                    print(f"[SUMMARY][{job_id}] erro doc {i}: {str(e)}")

            # -------------------------------
            # 🔥 VALIDAÇÃO FORTE
            # -------------------------------
            print(f"[SUMMARY][{job_id}] válidos: {len(partial_results)}")

            if not partial_results:
                raise Exception("Nenhuma evidência encontrada")

            # 🔥 limpa lixo comum
            filtered_results = [
                r for r in partial_results if "[]" not in r and "null" not in r.lower()
            ]

            print(f"[SUMMARY][{job_id}] após filtro: {len(filtered_results)}")

            if not filtered_results:
                raise Exception("Nenhuma evidência válida após filtro")

            # -------------------------------
            # 🔥 AGREGAÇÃO CONTROLADA
            # -------------------------------
            aggregated = "\n\n".join(filtered_results)

            print(f"[SUMMARY][{job_id}] Aggregated size: {len(aggregated)}")

            aggregated = aggregated[:20000]

            # -------------------------------
            # 🔥 PROMPT FINAL ANTI-HALLUCINATION
            # -------------------------------
            final_prompt = f"""
            {req.template}

            ====================
            BASE DE EVIDÊNCIAS
            ====================
            {aggregated}

            ====================
            REGRAS OBRIGATÓRIAS (CRÍTICAS)
            ====================

            1. TODA informação deve citar origem EXPLÍCITA

            Formato obrigatório:

            "FORMATO OBRIGATÓRIO:

            Toda afirmação deve seguir exatamente:

            "Dado identificado no documento: [NOME]
            Trecho:
            [COLAR TRECHO ORIGINAL]

            Análise:
            [explicação técnica]

            Resultado:
            [valor ou inconsistência]"

            Se não seguir esse formato → resposta inválida"

            2. PROIBIDO:
            - inventar valores
            - estimar números
            - generalizar sem evidência

            3. TODO dado deve conter:
            - documento de origem
            - trecho literal
            - explicação técnica

            4. Se não houver evidência suficiente:
            → escrever exatamente:
            "Dado não disponível nos arquivos analisados"

            5. NÃO produzir nenhuma afirmação sem citação

            ====================
            OBJETIVO
            ====================
            Gerar relatório executivo 100% rastreável.
            """

            job["stage"] = "final_llm"

            final = llm.invoke(final_prompt)

            job["result"] = final.content

        # -------------------------------
        job["status"] = "completed"
        job["stage"] = "done"

        print(f"[SUMMARY][{job_id}] ✅ DONE")

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)

        print(f"[SUMMARY][{job_id}] ❌ ERROR")
        print(traceback.format_exc())


# -------------------------------
# SUMMARY (COM LOGS 🔥)
# -------------------------------
@app.post("/generate-summary")
async def generate_summary(req: SummaryRequest):
    try:
        job_id = str(uuid.uuid4())

        summary_jobs[job_id] = {
            "status": "pending",
            "stage": "created",
            "project_id": req.project_id,
        }

        threading.Thread(target=process_summary_job, args=(job_id, req)).start()

        return {"job_id": job_id, "status": "started"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
