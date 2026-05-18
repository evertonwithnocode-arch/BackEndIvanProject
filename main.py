from dotenv import load_dotenv

load_dotenv()

import os
import uuid
import threading
import traceback
import shutil
import asyncio
import time
from typing import List, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import json, re
from supabase import create_client

# -------------------------------
# SUPABASE
# -------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise Exception("Credenciais do Supabase não encontradas")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

BUCKET_NAME = "sped-documents"

# -------------------------------
# JOB PERSISTENCE (backend_jobs)
# -------------------------------
JOBS_TABLE = "backend_jobs"


def job_create(job_id: str, kind: str, project_id: Optional[str] = None,
               stage: str = "created") -> None:
    supabase.table(JOBS_TABLE).insert({
        "id": job_id,
        "kind": kind,
        "project_id": project_id,
        "status": "pending",
        "stage": stage,
        "progress": 0,
    }).execute()


def job_update(job_id: str, **fields) -> None:
    allowed = {"status", "stage", "progress", "result", "error"}
    payload = {k: v for k, v in fields.items() if k in allowed}
    if not payload:
        return
    try:
        supabase.table(JOBS_TABLE).update(payload).eq("id", job_id).execute()
    except Exception as e:
        print(f"[job_update] erro {job_id}: {e}")


def job_get(job_id: str) -> Optional[dict]:
    try:
        res = supabase.table(JOBS_TABLE).select("*").eq("id", job_id).limit(1).execute()
        if res.data:
            return res.data[0]
        return None
    except Exception as e:
        print(f"[job_get] erro {job_id}: {e}")
        return None


def job_recover_stuck_on_startup() -> None:
    try:
        supabase.table(JOBS_TABLE).update({
            "status": "error",
            "error": "Servidor reiniciado durante o processamento. Reenvie a solicitação.",
        }).in_("status", ["pending", "processing"]).execute()
    except Exception as e:
        print(f"[job_recover_stuck_on_startup] {e}")


# -------------------------------
# FASTAPI APP
# -------------------------------
app = FastAPI()


@app.on_event("startup")
def _on_startup():
    job_recover_stuck_on_startup()


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
os.makedirs(PERSIST_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY não encontrada")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4.1", temperature=0.0, api_key=OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
)


# -------------------------------
# VECTOR STORE
# -------------------------------
def get_vector_store(project_id: str):
    try:
        project_path = os.path.join(PERSIST_DIR, project_id)
        return Chroma(
            collection_name=f"project_{project_id}",
            persist_directory=project_path,
            embedding_function=embeddings,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro vector store: {str(e)}")


# -------------------------------
# RAG HELPERS
# -------------------------------
def get_context(query: str, project_id: str, k: int = 10):
    try:
        vector_store = get_vector_store(project_id)
        docs = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=k * 4)

        if not docs:
            return "Nenhum dado encontrado para este projeto."

        context_parts = []
        for doc in docs:
            doc_type = doc.metadata.get("type", "document")
            prefix = "[ENRICHMENT]" if doc_type == "enrichment" else "[DOCUMENTO]"
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
        raise HTTPException(status_code=500, detail=f"Erro ao buscar contexto: {str(e)}")


def build_prompt(template: str, context: str, enrichment: dict | None):
    enrichment_text = ""
    if enrichment:
        enrichment_text = f"""
====================
DADOS DE ENRIQUECIMENTO
====================
{enrichment}
"""

    return f"""
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

3. Verificar base de cálculo, alíquota, valor do tributo e coerência entre eles.

4. Identificar erros como divergência total/itens, valores duplicados, CST incompatível, campos zerados.

5. Quando NÃO houver erro: dizer "Nenhuma inconsistência relevante encontrada".

6. Toda análise deve conter evidência (trecho real), explicação técnica e lógica aplicada.

7. Se fizer cálculo: mostrar fórmula, valores usados e resultado.

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
"""


# -------------------------------
# MODELS
# -------------------------------
class SummaryRequest(BaseModel):
    template: str
    query: Optional[str] = "gerar sumário geral"
    enrichment: Optional[Dict] = None
    k: Optional[int] = 20
    project_id: str


class EnrichmentRequest(BaseModel):
    project_id: str
    enrichment: Dict
    source: Optional[str] = "manual_enrichment"


# -------------------------------
# PROCESS JOB (chunking + embedding)
# -------------------------------
def process_job(job_id: str, files_data: List[dict], project_id: str):
    try:
        job_update(job_id, status="processing", stage="chunking", progress=0)

        vector_store = get_vector_store(project_id)
        all_chunks = []
        all_metadata = []

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
                all_metadata.append({
                    "source": filename,
                    "chunk_index": idx,
                    "project_id": project_id,
                    "type": "document",
                })

        job_update(job_id, progress=50, stage="embedding")

        BATCH_SIZE = 100
        num_chunks = len(all_chunks)
        num_batches = (num_chunks + BATCH_SIZE - 1) // BATCH_SIZE

        def add_batch(batch_idx):
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    start = batch_idx * BATCH_SIZE
                    end = min(start + BATCH_SIZE, num_chunks)
                    vector_store.add_texts(
                        texts=all_chunks[start:end],
                        metadatas=all_metadata[start:end],
                    )
                    return batch_idx
                except Exception as e:
                    if "429" in str(e) and attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    raise e

        with ThreadPoolExecutor(max_workers=1) as t_executor:
            futures = []
            for i in range(num_batches):
                futures.append(t_executor.submit(add_batch, i))
                time.sleep(0.2)

            for idx, future in enumerate(futures):
                future.result()
                progress = 50 + int(((idx + 1) / num_batches) * 50)
                job_update(job_id, progress=progress)

        job_update(
            job_id,
            status="completed",
            stage="done",
            progress=100,
            result={
                "total_chunks": num_chunks,
                "total_files": len(files_data),
            },
        )

    except Exception as e:
        print(f"Erro Crítico no Job {job_id}:")
        print(traceback.format_exc())
        job_update(job_id, status="error", error=str(e))


# -------------------------------
# PROCESS SUMMARY JOB
# -------------------------------
def process_summary_job(job_id: str, req: SummaryRequest):
    try:
        job_update(job_id, status="processing", stage="starting")
        print(f"[SUMMARY][{job_id}] 🚀 START")

        mode = "strategic" if "DOCUMENTO 1" in req.template else "audit"
        print(f"[SUMMARY][{job_id}] Mode: {mode}")

        # ===============================
        # 🔵 AUDIT MODE
        # ===============================
        if mode == "audit":
            job_update(job_id, stage="retrieving_context")
            context = get_context(req.query, req.project_id, req.k)
            print(f"[SUMMARY][{job_id}] Context size: {len(context)}")
            context = context[:12000]

            job_update(job_id, stage="building_prompt")
            prompt = build_prompt(req.template, context, req.enrichment)
            print(f"[SUMMARY][{job_id}] Prompt size: {len(prompt)}")

            job_update(job_id, stage="llm_call")
            response = llm.invoke(prompt)

            job_update(
                job_id,
                status="completed",
                stage="done",
                result={"mode": mode, "content": response.content},
            )

        # ===============================
        # 🔴 STRATEGIC MODE
        # ===============================
        else:
            job_update(job_id, stage="multi_step_rag")
            vector_store = get_vector_store(req.project_id)

            docs = vector_store.max_marginal_relevance_search(
                req.query, k=20, fetch_k=40
            )
            print(f"[SUMMARY][{job_id}] Docs encontrados: {len(docs)}")

            partial_results = []
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
                    if content and "evidencias" in content:
                        partial_results.append(content)
                except Exception as e:
                    print(f"[SUMMARY][{job_id}] erro doc {i}: {str(e)}")

            print(f"[SUMMARY][{job_id}] válidos: {len(partial_results)}")
            if not partial_results:
                raise Exception("Nenhuma evidência encontrada")

            filtered_results = []

            for idx, r in enumerate(partial_results, start=1):
                print(f"[SUMMARY][{job_id}] FILTER CHECK {idx}")

                match = re.search(r"\{.*\}", r, re.DOTALL)

                if not match:
                    print(f"[SUMMARY][{job_id}] FILTER {idx}: sem JSON detectado")
                    print(r[:1000])
                    continue

                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError as e:
                    print(f"[SUMMARY][{job_id}] FILTER {idx}: JSON inválido: {str(e)}")
                    print(match.group(0)[:1000])
                    continue

                evidencias = data.get("evidencias") or []
                print(f"[SUMMARY][{job_id}] FILTER {idx}: evidencias={len(evidencias)}")

                valid_evidencias = []

                for e_idx, e in enumerate(evidencias, start=1):
                    trecho = e.get("trecho")
                    documento = e.get("documento")
                    registro = e.get("registro")

                    print(
                        f"[SUMMARY][{job_id}] FILTER {idx}.{e_idx}: "
                                    f"documento={documento!r}, registro={registro!r}, "
                        f"trecho_len={len(str(trecho or ''))}"
                    )

                    if trecho and str(trecho).strip():
                        valid_evidencias.append(e)

                print(f"[SUMMARY][{job_id}] FILTER {idx}: valid_evidencias={len(valid_evidencias)}")

                if valid_evidencias:
                    data["evidencias"] = valid_evidencias
                    filtered_results.append(json.dumps(data, ensure_ascii=False))


            print(f"[SUMMARY][{job_id}] após filtro: {len(filtered_results)}")

            if not filtered_results:
             print(f"[SUMMARY][{job_id}] ⚠️ Nenhuma evidência estruturada encontrada")
             print(f"[SUMMARY][{job_id}] ⚠️ Usando fallback com respostas brutas")

             fallback_results = [
                 r for r in partial_results
                 if r and isinstance(r, str) and len(r.strip()) > 20
             ]

             if not fallback_results:
                 raise Exception("Nenhuma resposta útil retornada pelo LLM")

             filtered_results = fallback_results

            aggregated = "\n\n".join(filtered_results)
            print(f"[SUMMARY][{job_id}] Aggregated size: {len(aggregated)}")
            aggregated = aggregated[:20000]

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

            "Dado identificado no documento: [NOME]
            Trecho:
            [COLAR TRECHO ORIGINAL]

            Análise:
            [explicação técnica]

            Resultado:
            [valor ou inconsistência]"

            2. PROIBIDO inventar valores, estimar números, generalizar sem evidência.
            3. TODO dado deve conter documento de origem, trecho literal e explicação técnica.
            4. Se não houver evidência suficiente → "Dado não disponível nos arquivos analisados"
            5. NÃO produzir nenhuma afirmação sem citação.
            """

            job_update(job_id, stage="final_llm")
            final = llm.invoke(final_prompt)

            job_update(
                job_id,
                status="completed",
                stage="done",
                result={"mode": mode, "content": final.content},
            )

        print(f"[SUMMARY][{job_id}] ✅ DONE")

    except Exception as e:
        print(f"[SUMMARY][{job_id}] ❌ ERROR")
        print(traceback.format_exc())
        job_update(job_id, status="error", error=str(e))


# -------------------------------
# ENDPOINTS
# -------------------------------
@app.post("/enrichment")
async def upload_enrichment(req: EnrichmentRequest):
    try:
        vector_store = get_vector_store(req.project_id)
        enrichment_text = enrichment_to_text(req.enrichment)
        chunks = text_splitter.split_text(enrichment_text)

        texts, metadatas = [], []
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "project_id": req.project_id,
                "type": "enrichment",
                "source": req.source,
                "chunk_index": idx,
            })

        vector_store.add_texts(texts=texts, metadatas=metadatas)
        return {"status": "success", "chunks_saved": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


@app.post("/upload")
async def upload_documents(project_id: str, files: List[UploadFile] = File(...)):
    try:
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id obrigatório")

        job_id = str(uuid.uuid4())

        async def process_single_file(file: UploadFile):
            content = await file.read()
            text = content.decode("utf-8", errors="ignore")
            if text.strip():
                return {"filename": file.filename, "text": text}
            return None

        results = await asyncio.gather(*[process_single_file(f) for f in files])
        files_data = [res for res in results if res is not None]

        if not files_data:
            raise HTTPException(status_code=400, detail="Nenhum arquivo válido")

        # Persiste o job no Supabase
        job_create(job_id, kind="process", project_id=project_id)

        # Dispara worker em background
        threading.Thread(
            target=process_job, args=(job_id, files_data, project_id)
        ).start()

        return {"job_id": job_id, "project_id": project_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-project/{project_id}")
def delete_project(project_id: str, folder_path: str):
    try:
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id obrigatório")
        if not folder_path:
            raise HTTPException(status_code=400, detail="folder_path obrigatório")

        try:
            while True:
                files = supabase.storage.from_(BUCKET_NAME).list(path=folder_path)
                if not files:
                    break
                paths = [f"{folder_path.rstrip('/')}/{file['name']}" for file in files]
                supabase.storage.from_(BUCKET_NAME).remove(paths)
                print(f"🗑️ Deletados {len(paths)} arquivos...")
                time.sleep(0.2)
        except Exception as e:
            print("⚠️ Erro ao deletar storage:", str(e))

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
        raise HTTPException(status_code=500, detail=f"Erro ao deletar projeto: {str(e)}")


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job não encontrado")
    return job


@app.get("/summary-status/{job_id}")
def get_summary_status(job_id: str):
    job = job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Summary job não encontrado")
    return job


@app.post("/generate-summary")
async def generate_summary(req: SummaryRequest):
    try:
        job_id = str(uuid.uuid4())

        # Persiste o job no Supabase
        job_create(job_id, kind="summary", project_id=req.project_id)

        # Dispara worker em background
        threading.Thread(target=process_summary_job, args=(job_id, req)).start()

        return {"job_id": job_id, "status": "started"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
