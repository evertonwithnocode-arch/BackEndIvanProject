from dotenv import load_dotenv

load_dotenv()

import os
import uuid
import threading
import traceback
import shutil
import asyncio
import time
from collections import Counter
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


def job_create(job_id, kind, project_id=None, stage="created"):
    supabase.table(JOBS_TABLE).insert({
        "id": job_id, "kind": kind, "project_id": project_id,
        "status": "pending", "stage": stage, "progress": 0,
    }).execute()


def job_update(job_id, **fields):
    allowed = {"status", "stage", "progress", "result", "error"}
    payload = {k: v for k, v in fields.items() if k in allowed}
    if not payload:
        return
    try:
        supabase.table(JOBS_TABLE).update(payload).eq("id", job_id).execute()
    except Exception as e:
        print(f"[job_update] erro {job_id}: {e}")


def job_get(job_id):
    try:
        res = supabase.table(JOBS_TABLE).select("*").eq("id", job_id).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"[job_get] erro {job_id}: {e}")
        return None


def job_recover_stuck_on_startup():
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
    if request.headers.get("x-api-key") != INTERNAL_API_KEY:
        return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
    return await call_next(request)


app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
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

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-5", temperature=0.0, api_key=OPENAI_API_KEY)

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


# >>> ALTERAÇÃO #2: detector de registro SPED a partir do texto do chunk.
# Identifica blocos no início da linha (|0450|, |C170|, |M200|, etc.)
SPED_REGISTRO_REGEX = re.compile(r"^\|([0-9A-Z]{4})\|", re.MULTILINE)

def detect_registros(text: str) -> List[str]:
    """Retorna lista única de registros SPED encontrados no chunk (ex.: ['0450','C170'])."""
    if not text:
        return []
    found = SPED_REGISTRO_REGEX.findall(text)
    # preserva ordem mas remove duplicados
    seen, out = set(), []
    for r in found:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def primary_registro(text: str) -> Optional[str]:
    """Registro 'dominante' do chunk (o que mais aparece)."""
    if not text:
        return None
    matches = SPED_REGISTRO_REGEX.findall(text)
    if not matches:
        return None
    return Counter(matches).most_common(1)[0][0]


# >>> ALTERAÇÃO #1: limite máximo de chunks 0450 no contexto final
MAX_0450_CHUNKS = 2


def cap_registro_chunks(docs, registro: str, max_keep: int):
    """Mantém no máximo `max_keep` chunks cujo registro primário == registro."""
    kept, dropped = [], 0
    count = 0
    for d in docs:
        reg = d.metadata.get("registro") or primary_registro(d.page_content)
        if reg == registro:
            if count < max_keep:
                kept.append(d)
                count += 1
            else:
                dropped += 1
        else:
            kept.append(d)
    return kept, dropped


# -------------------------------
# RAG HELPERS
# -------------------------------
def get_context(query: str, project_id: str, k: int = 10):
    try:
        vector_store = get_vector_store(project_id)
        docs = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=k * 4)

        if not docs:
            return "Nenhum dado encontrado para este projeto."

        # >>> ALTERAÇÃO #1 (audit mode): aplicar cap em 0450 também aqui
        docs, dropped = cap_registro_chunks(docs, "0450", MAX_0450_CHUNKS)
        if dropped:
            print(f"[get_context] descartados {dropped} chunks 0450 acima do limite")

        context_parts = []
        for doc in docs:
            doc_type = doc.metadata.get("type", "document")
            prefix = "[ENRICHMENT]" if doc_type == "enrichment" else "[DOCUMENTO]"
            reg = doc.metadata.get("registro") or primary_registro(doc.page_content) or "?"
            context_parts.append(
                f"""
{prefix}
Fonte: {doc.metadata.get("source")}
Chunk: {doc.metadata.get("chunk_index")}
Registro: {reg}

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
2. Sempre validar consistência entre registros (A100 vs A170, totais vs itens, valores divergentes).
3. Verificar base de cálculo, alíquota, valor do tributo.
4. Identificar erros: divergência total/itens, valores duplicados, CST incompatível, campos zerados.
5. Quando NÃO houver erro: "Nenhuma inconsistência relevante encontrada".
6. Toda análise deve conter evidência (trecho real), explicação técnica e lógica aplicada.
7. Se fizer cálculo: mostrar fórmula, valores usados e resultado.

# >>> ALTERAÇÃO #4: regras de formatação obrigatórias do output
8. NÃO use o registro 0450 (informações complementares) como evidência principal de análise
   financeira. Ele é apenas dicionário de observações — priorize blocos com valores:
   M100, M200, M500, M600 (apuração) e C100, C170, C190, C500, D100 (documentos/itens).
9. Ao numerar seções, substitua qualquer placeholder `X.N` pelo número real do capítulo.
   NUNCA escreva `X.` literal no texto final (ex.: use "4.3" e não "X.3" ou ".3").
10. Se um item realmente não tiver dado nos arquivos analisados, escreva
    "Dado não disponível nos arquivos analisados" SOMENTE uma vez por seção — não duplique.

{{
  "insights": [
    {{"titulo": "", "explicacao": "", "passo_a_passo": [],
      "dados_utilizados": [], "logica_aplicada": "", "conclusao": ""}}
  ],
  "inconsistencias": [
    {{"titulo": "", "descricao": "", "impacto": "",
      "evidencias": [{{"fonte": "", "trecho": ""}}], "recomendacao": ""}}
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
        all_chunks, all_metadata = [], []

        local_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        with ProcessPoolExecutor() as executor:
            texts = [f["text"] for f in files_data]
            results = list(executor.map(local_splitter.split_text, texts))

        for i, chunks in enumerate(results):
            filename = files_data[i]["filename"]
            for idx, chunk in enumerate(chunks):
                # >>> ALTERAÇÃO #2: tag de registro SPED na metadata
                reg = primary_registro(chunk)
                regs_all = detect_registros(chunk)
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": filename,
                    "chunk_index": idx,
                    "project_id": project_id,
                    "type": "document",
                    "registro": reg or "unknown",
                    "registros_all": ",".join(regs_all) if regs_all else "",
                })

        job_update(job_id, progress=50, stage="embedding")

        BATCH_SIZE = 100
        num_chunks = len(all_chunks)
        num_batches = (num_chunks + BATCH_SIZE - 1) // BATCH_SIZE

        def add_batch(batch_idx):
            for attempt in range(3):
                try:
                    start = batch_idx * BATCH_SIZE
                    end = min(start + BATCH_SIZE, num_chunks)
                    vector_store.add_texts(
                        texts=all_chunks[start:end],
                        metadatas=all_metadata[start:end],
                    )
                    return batch_idx
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        time.sleep(2 * (attempt + 1))
                        continue
                    raise

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
            job_id, status="completed", stage="done", progress=100,
            result={"total_chunks": num_chunks, "total_files": len(files_data)},
        )

    except Exception as e:
        print(f"Erro Crítico no Job {job_id}:")
        print(traceback.format_exc())
        job_update(job_id, status="error", error=str(e))


# >>> ALTERAÇÃO #3: queries dirigidas a blocos diferentes do SPED.
# Em vez de só `req.query`, fazemos múltiplas buscas e unimos os resultados.
MULTI_QUERIES = [
    "apuração de PIS COFINS, base de cálculo, alíquota e valor do tributo (registros M100 M200 M500 M600)",
    "documentos fiscais, itens, NCM, CST, CFOP, valor da operação (C100 C170 C190 C500 D100)",
    "totais consolidados, ajustes e créditos do período",
    "inconsistências, divergências entre totais e itens, valores zerados ou duplicados",
]


def multi_query_retrieval(vector_store, base_query: str, per_query_k: int = 8):
    """Executa várias queries e deduplica por (source, chunk_index)."""
    queries = [base_query] + MULTI_QUERIES
    seen, merged = set(), []
    for q in queries:
        try:
            docs = vector_store.max_marginal_relevance_search(
                q, k=per_query_k, fetch_k=per_query_k * 3,
                filter={"type": "document"},
            )
        except Exception as e:
            print(f"[multi_query_retrieval] falhou query={q[:40]!r}: {e}")
            continue
        for d in docs:
            key = (d.metadata.get("source"), d.metadata.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(d)
    return merged


# >>> ALTERAÇÃO #5: log de distribuição de registros
def log_registro_distribution(job_id: str, docs, label: str):
    dist = Counter()
    for d in docs:
        reg = d.metadata.get("registro") or primary_registro(d.page_content) or "unknown"
        dist[reg] += 1
    pretty = ", ".join(f"{k}={v}" for k, v in dist.most_common())
    print(f"[SUMMARY][{job_id}] DIST {label}: total={len(docs)} | {pretty}")
    return dist


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

            job_update(job_id, status="completed", stage="done",
                       result={"mode": mode, "content": response.content})

        # ===============================
        # 🔴 STRATEGIC MODE
        # ===============================
        else:
            job_update(job_id, stage="multi_step_rag")
            vector_store = get_vector_store(req.project_id)

            # >>> ALTERAÇÃO #3: multi-query retrieval (em vez de só req.query)
            docs = multi_query_retrieval(vector_store, req.query, per_query_k=8)
            print(f"[SUMMARY][{job_id}] Docs após multi-query: {len(docs)}")

            # Fallback defensivo
            if not docs:
                print(f"[SUMMARY][{job_id}] ⚠️ multi-query vazio, fallback simples")
                docs = vector_store.max_marginal_relevance_search(
                    req.query, k=20, fetch_k=40
                )

            # >>> ALTERAÇÃO #5: distribuição ANTES do cap
            log_registro_distribution(job_id, docs, "pré-cap")

            # >>> ALTERAÇÃO #1: limitar 0450 no contexto
            docs, dropped = cap_registro_chunks(docs, "0450", MAX_0450_CHUNKS)
            print(f"[SUMMARY][{job_id}] 0450 descartados: {dropped}")

            # >>> ALTERAÇÃO #5: distribuição APÓS o cap
            log_registro_distribution(job_id, docs, "pós-cap")

            # Log detalhado
            for i, doc in enumerate(docs, start=1):
                src = doc.metadata.get("source", "?")
                reg = doc.metadata.get("registro") or primary_registro(doc.page_content) or "?"
                print(f"[SUMMARY][{job_id}] Doc {i}/{len(docs)} reg={reg} source={src}")

            partial_results = []
            for i, doc in enumerate(docs):
                reg = doc.metadata.get("registro") or primary_registro(doc.page_content) or "?"
                prompt = f"""
                Extraia APENAS dados REAIS do texto.
                Registro SPED predominante: {reg}

                RETORNE JSON:
                {{
                "evidencias": [
                    {{
                      "documento": "{doc.metadata.get("source")}",
                      "chunk": "{doc.metadata.get("chunk_index")}",
                      "registro": "{reg}",
                      "trecho": "",
                      "tipo": "financeiro | inconsistencia | fiscal | cadastral"
                    }}
                  ]
                }}

                REGRAS:
                - NÃO inventar nada
                - NÃO resumir
                - NÃO estimar valores
                - Se o registro for 0450 (informação complementar), marque tipo="cadastral"
                  e NÃO o trate como evidência financeira.
                - Se não houver evidência → retornar lista vazia

                TEXTO:
                {doc.page_content}
                """
                try:
                    res = llm.invoke(prompt)
                    content = res.content.strip()
                    if content and "evidencias" in content:
                        partial_results.append((content, doc))
                except Exception as e:
                    print(f"[SUMMARY][{job_id}] erro doc {i}: {str(e)}")

            print(f"[SUMMARY][{job_id}] válidos: {len(partial_results)}")
            if not partial_results:
                raise Exception("Nenhuma evidência encontrada")

            filtered_results = []
            for idx, (r, doc) in enumerate(partial_results, start=1):
                match = re.search(r"\{.*\}", r, re.DOTALL)
                if not match:
                    continue
                try:
                    data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    continue

                evidencias = data.get("evidencias") or []

                if not evidencias:
                    print(f"[SUMMARY][{job_id}] ⚠️ FILTER {idx}: fallback bruto")
                    evidencias = [{
                        "documento": doc.metadata.get("source", "arquivo_sped"),
                        "chunk": doc.metadata.get("chunk_index"),
                        "registro": doc.metadata.get("registro", "?"),
                        "trecho": doc.page_content[:1200],
                        "tipo": "sped_raw",
                    }]

                valid = [e for e in evidencias if e.get("trecho") and str(e["trecho"]).strip()]
                if valid:
                    data["evidencias"] = valid
                    filtered_results.append(json.dumps(data, ensure_ascii=False))

            print(f"[SUMMARY][{job_id}] após filtro: {len(filtered_results)}")

            if not filtered_results:
                fallback = [r for (r, _) in partial_results if r and len(r.strip()) > 20]
                if not fallback:
                    raise Exception("Nenhuma resposta útil retornada pelo LLM")
                filtered_results = fallback

            aggregated = "\n\n".join(filtered_results)[:20000]
            print(f"[SUMMARY][{job_id}] Aggregated size: {len(aggregated)}")

            enrichment_block = ""
            if req.enrichment:
                enrichment_block = f"""
            ====================
            DADOS DE ENRIQUECIMENTO (CADASTRAL — COMPLEMENTAR)
            ====================
            {req.enrichment}
            """

            # >>> ALTERAÇÃO #4: regras de formatação no prompt final
            final_prompt = f"""
            {req.template}

            ====================
            BASE DE EVIDÊNCIAS
            ====================
            {aggregated}
            {enrichment_block}
            ====================
            REGRAS OBRIGATÓRIAS (CRÍTICAS)
            ====================
            1. TODA informação deve citar origem EXPLÍCITA (documento + trecho literal).
            2. PROIBIDO inventar valores, estimar números, generalizar sem evidência.
            3. Se não houver evidência → "Dado não disponível nos arquivos analisados"
               (apenas UMA vez por seção, não repita).
            4. NÃO produzir nenhuma afirmação sem citação.

            REGRAS DE FORMATAÇÃO DO OUTPUT:
            5. Ao numerar seções, SUBSTITUA qualquer placeholder do tipo `X.N` pelo número
               real do capítulo. NUNCA escreva `X.`, `.3`, `X.3` literais no texto final.
               Exemplo correto: "4.3 Cálculo do Impacto Financeiro" (não "X.3").
            6. NÃO duplique títulos de seção (ex.: não repita "X.3 Cálculo do Impacto..." 3x).
            7. Registros 0450 são INFORMAÇÕES COMPLEMENTARES (dicionário de observações).
               NÃO os use como base para cálculo de impacto financeiro ou ROI — esses cálculos
               devem vir de M100/M200/M500/M600 e itens C170. Se faltar essa base, declare
               ausência explicitamente em vez de citar 0450.
            """

            job_update(job_id, stage="final_llm")
            final = llm.invoke(final_prompt)

            job_update(job_id, status="completed", stage="done",
                       result={"mode": mode, "content": final.content})

        print(f"[SUMMARY][{job_id}] ✅ DONE")

    except Exception as e:
        print(f"[SUMMARY][{job_id}] ❌ ERROR")
        print(traceback.format_exc())
        job_update(job_id, status="error", error=str(e))


# -------------------------------
# ENDPOINTS  (inalterados)
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
                "project_id": req.project_id, "type": "enrichment",
                "source": req.source, "chunk_index": idx,
                "registro": "enrichment",
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

        job_create(job_id, kind="process", project_id=project_id)
        threading.Thread(target=process_job, args=(job_id, files_data, project_id)).start()
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
                time.sleep(0.2)
        except Exception as e:
            print("⚠️ Erro storage:", str(e))

        project_path = os.path.join(PERSIST_DIR, project_id)
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
        return {"status": "success", "message": f"Project {project_id} deletado"}
    except Exception as e:
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
        job_create(job_id, kind="summary", project_id=req.project_id)
        threading.Thread(target=process_summary_job, args=(job_id, req)).start()
        return {"job_id": job_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
