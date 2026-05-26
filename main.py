import re
import json
from openai import RateLimitError, APIError, AuthenticationError, BadRequestError
from typing import List
import logging
from openai import OpenAI
from supabase import create_client
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.callbacks.manager import get_openai_callback
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Optional, Dict, Tuple, Any
from collections import Counter
import time
import asyncio
import shutil
import traceback
import threading
import uuid
import os
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)

MAX_CHARS_PER_CHUNK = 24000
EMPTY_PLACEHOLDER = " "


def _sanitize_texts(texts: List[str]) -> List[str]:
    cleaned: List[str] = []
    for t in texts:
        if t is None:
            cleaned.append(EMPTY_PLACEHOLDER)
            continue
        s = str(t).strip()
        if not s:
            s = EMPTY_PLACEHOLDER
        if len(s) > MAX_CHARS_PER_CHUNK:
            s = s[:MAX_CHARS_PER_CHUNK]
        cleaned.append(s)
    return cleaned


# -------------------------------
# DECODER INTELIGENTE PARA SPED
# -------------------------------
def smart_decode_sped(raw: bytes) -> str:
    if not raw:
        return ""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    try:
        decoded = raw.decode("latin-1")
        if any(c in decoded for c in ("\x80", "\x82", "\x83", "\x84", "\x85", "\x86", "\x87", "\x88", "\x89")):
            try:
                return raw.decode("cp1252")
            except UnicodeDecodeError:
                return decoded
        return decoded
    except UnicodeDecodeError:
        pass
    return raw.decode("utf-8", errors="replace")


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
# JOB PERSISTENCE
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
CHUNK_OVERLAP = 150
PERSIST_DIR = "/data/chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY não encontrada")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 256
LLM_MODEL = "gpt-4.1"

# -------------------------------
# AGENTIC RAG CONFIG
# -------------------------------
AGENTIC_MAX_ITERATIONS = 15           # nº máximo de "passos" do agente (search/think/write)
AGENTIC_MAX_SEARCHES = 12             # nº máximo de buscas distintas no Chroma
AGENTIC_DEFAULT_K = 10                # k default por busca quando agente não especifica
AGENTIC_MAX_K = 25                    # teto de k por busca
AGENTIC_MAX_CHUNK_CHARS = 1800        # corte por chunk antes de devolver para o LLM
AGENTIC_TOOL_TIMEOUT_S = 30           # timeout por chamada ao Chroma


class BatchOpenAIEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        safe_texts = _sanitize_texts(texts)
        out: List[List[float]] = []
        for i in range(0, len(safe_texts), EMBED_BATCH):
            batch = safe_texts[i:i + EMBED_BATCH]
            embeddings_batch: List[List[float]] = []
            for attempt in range(5):
                try:
                    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
                    embeddings_batch = [d.embedding for d in resp.data]
                    if len(embeddings_batch) != len(batch):
                        raise RuntimeError(
                            f"OpenAI retornou {len(embeddings_batch)} embeddings para {len(batch)} inputs."
                        )
                    break
                except AuthenticationError as e:
                    logger.error("OpenAI auth error: %s", e)
                    raise RuntimeError("OpenAI API Key inválida ou revogada.") from e
                except RateLimitError as e:
                    msg = str(e)
                    if "insufficient_quota" in msg or "exceeded your current quota" in msg:
                        raise RuntimeError("OpenAI sem créditos disponíveis.") from e
                    if attempt < 4:
                        time.sleep(2 ** attempt); continue
                    raise
                except BadRequestError as e:
                    logger.error("OpenAI bad request: %s", e); raise
                except APIError as e:
                    if attempt < 4:
                        time.sleep(2 ** attempt); continue
                    raise
                except Exception as e:
                    msg = str(e)
                    if ("429" in msg or "rate" in msg.lower()) and attempt < 4:
                        time.sleep(2 ** attempt); continue
                    raise
            if len(embeddings_batch) != len(batch):
                raise RuntimeError(f"Falha ao gerar embeddings batch {i}-{i+len(batch)}.")
            out.extend(embeddings_batch)
        if len(out) != len(texts):
            raise RuntimeError(f"Inconsistência: {len(out)} embeddings para {len(texts)} textos.")
        return out

    def embed_query(self, text: str) -> List[float]:
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
        return resp.data[0].embedding


embeddings = BatchOpenAIEmbeddings()
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0, api_key=OPENAI_API_KEY)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


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
# SPED helpers
# -------------------------------
SPED_REGISTRO_REGEX = re.compile(r"^\|([0-9A-Z]{4})\|", re.MULTILINE)
SPED_0000_REGEX = re.compile(r"\|0000\|[^|]*\|[^|]*\|(\d{8})\|(\d{8})\|", re.MULTILINE)


def detect_registros(text: str) -> List[str]:
    if not text:
        return []
    found = SPED_REGISTRO_REGEX.findall(text)
    seen, out = set(), []
    for r in found:
        if r not in seen:
            seen.add(r); out.append(r)
    return out


def primary_registro(text: str) -> Optional[str]:
    if not text:
        return None
    matches = SPED_REGISTRO_REGEX.findall(text)
    if not matches:
        return None
    return Counter(matches).most_common(1)[0][0]


def extract_periodo_from_docs(docs) -> Optional[str]:
    dt_inis, dt_fins = [], []
    for d in docs:
        text = getattr(d, "page_content", "") or ""
        for m in SPED_0000_REGEX.finditer(text):
            dt_ini_raw, dt_fin_raw = m.group(1), m.group(2)
            try:
                dt_inis.append(f"{dt_ini_raw[0:2]}/{dt_ini_raw[2:4]}/{dt_ini_raw[4:8]}")
                dt_fins.append(f"{dt_fin_raw[0:2]}/{dt_fin_raw[2:4]}/{dt_fin_raw[4:8]}")
            except Exception:
                continue
    if not dt_inis or not dt_fins:
        return None
    sorted_ini = sorted(dt_inis, key=lambda x: (x[6:10], x[3:5], x[0:2]))
    sorted_fin = sorted(dt_fins, key=lambda x: (x[6:10], x[3:5], x[0:2]))
    return f"{sorted_ini[0]} a {sorted_fin[-1]}"


MAX_0450_CHUNKS = 2


def cap_registro_chunks(docs, registro: str, max_keep: int):
    kept, dropped, count = [], 0, 0
    for d in docs:
        reg = d.metadata.get("registro") or primary_registro(d.page_content)
        if reg == registro:
            if count < max_keep:
                kept.append(d); count += 1
            else:
                dropped += 1
        else:
            kept.append(d)
    return kept, dropped


# -------------------------------
# DRIVA PRE-PROCESSING
# -------------------------------
DRIVA_RELEVANT_KEYS = {
    "cnpj", "razao_social", "nome_fantasia", "natureza_juridica",
    "porte", "porte_empresa", "capital_social", "data_abertura", "data_inicio_atividade",
    "situacao_cadastral", "situacao", "motivo_situacao", "regime_tributario",
    "simples_nacional", "mei", "opcao_simples", "opcao_mei",
    "cnae_principal", "cnae_fiscal", "cnae_fiscal_descricao",
    "cnaes_secundarios", "atividade_principal", "atividades_secundarias",
    "endereco", "municipio", "uf", "cep",
    "telefone", "email",
    "qsa", "socios", "quadro_socios",
    "matriz_filial", "tipo",
}


def _filter_driva_dict(data: Any, depth: int = 0) -> Any:
    if depth > 6:
        return None
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            k_low = str(k).lower().strip()
            if k_low in DRIVA_RELEVANT_KEYS or any(rel in k_low for rel in ("cnae", "socio", "regime", "tribut", "porte", "situacao")):
                filtered = _filter_driva_dict(v, depth + 1)
                if filtered not in (None, "", [], {}):
                    out[k] = filtered
        return out
    if isinstance(data, list):
        out_list = []
        for item in data[:10]:
            filtered = _filter_driva_dict(item, depth + 1)
            if filtered not in (None, "", [], {}):
                out_list.append(filtered)
        return out_list
    return data


def preprocess_driva(enrichment: Dict) -> Dict:
    if not isinstance(enrichment, dict):
        return {}
    filtered = _filter_driva_dict(enrichment)
    return filtered if isinstance(filtered, dict) else {}


# -------------------------------
# RAG FILTERS / HELPERS (one-shot legados)
# -------------------------------
def _filter_sped() -> Dict:
    return {"source_kind": "sped"}


def _filter_driva() -> Dict:
    return {"source_kind": "driva"}


def get_context(query: str, project_id: str, k: int = 10):
    try:
        vector_store = get_vector_store(project_id)
        try:
            sped_docs = vector_store.max_marginal_relevance_search(
                query, k=k, fetch_k=k * 4, filter=_filter_sped()
            )
        except Exception:
            sped_docs = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=k * 4)

        if not sped_docs:
            return "Nenhum dado SPED encontrado para este projeto."

        sped_docs, dropped = cap_registro_chunks(sped_docs, "0450", MAX_0450_CHUNKS)
        if dropped:
            print(f"[get_context] descartados {dropped} chunks 0450")

        sped_parts = []
        for doc in sped_docs:
            reg = doc.metadata.get("registro") or primary_registro(doc.page_content) or "?"
            sped_parts.append(
                f"[SPED] Fonte: {doc.metadata.get('source')} | Chunk: {doc.metadata.get('chunk_index')} | Registro: {reg}\n{doc.page_content}"
            )

        driva_parts = []
        try:
            driva_docs = vector_store.max_marginal_relevance_search(
                "porte regime tributário CNAE atividade sócios capital",
                k=4, fetch_k=12, filter=_filter_driva(),
            )
            for doc in driva_docs:
                driva_parts.append(
                    f"[DRIVA] Fonte: {doc.metadata.get('source')} | Chunk: {doc.metadata.get('chunk_index')}\n{doc.page_content}"
                )
        except Exception as e:
            print(f"[get_context] driva retrieval falhou: {e}")

        result = "## EVIDÊNCIAS FISCAIS (SPED) — fonte primária\n\n" + "\n\n".join(sped_parts)
        if driva_parts:
            result += "\n\n## CONTEXTO DE NEGÓCIO (Driva) — apoio, NÃO usar como evidência numérica\n\n" + "\n\n".join(driva_parts)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar contexto: {str(e)}")


def driva_context_retrieval(vector_store, k: int = 5) -> List[Any]:
    try:
        return vector_store.max_marginal_relevance_search(
            "porte regime tributário CNAE atividade sócios capital situação cadastral",
            k=k, fetch_k=k * 3, filter=_filter_driva(),
        )
    except Exception as e:
        print(f"[driva_context_retrieval] falhou: {e}")
        return []


# -------------------------------
# MODELS
# -------------------------------
class SummaryRequest(BaseModel):
    template: str
    query: Optional[str] = "gerar sumário geral"
    enrichment: Optional[Dict] = None  # DEPRECATED: indexar via /enrichment
    k: Optional[int] = 20
    project_id: str
    # 🆕 modo agêntico: se True, usa loop tool-calling em vez de one-shot RAG
    agentic: Optional[bool] = True


class EnrichmentRequest(BaseModel):
    project_id: str
    enrichment: Dict
    source: Optional[str] = "manual_enrichment"


class ProcessPathsRequest(BaseModel):
    project_id: str
    paths: List[str]


# -------------------------------
# PROCESS JOB (SPED) — indexação
# -------------------------------
def process_job(job_id: str, files_data: List[dict], project_id: str):
    try:
        t0 = time.time()
        job_update(job_id, status="processing", stage="chunking", progress=5)
        vector_store = get_vector_store(project_id)
        all_chunks, all_metadata = [], []
        local_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        with ProcessPoolExecutor() as executor:
            texts = [f["text"] for f in files_data]
            results = list(executor.map(local_splitter.split_text, texts))

        for i, chunks in enumerate(results):
            filename = files_data[i]["filename"]
            for idx, chunk in enumerate(chunks):
                reg = primary_registro(chunk)
                regs_all = detect_registros(chunk)
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": filename,
                    "source_kind": "sped",
                    "chunk_index": idx,
                    "project_id": project_id,
                    "type": "document",
                    "registro": reg or "unknown",
                    "registros_all": ",".join(regs_all) if regs_all else "",
                })

        num_chunks = len(all_chunks)
        print(f"[JOB {job_id}] chunking ok: {num_chunks} chunks em {time.time()-t0:.1f}s")
        job_update(job_id, progress=30, stage="embedding")

        INSERT_BATCH = 512
        num_batches = (num_chunks + INSERT_BATCH - 1) // INSERT_BATCH
        for b in range(num_batches):
            start = b * INSERT_BATCH
            end = min(start + INSERT_BATCH, num_chunks)
            for attempt in range(3):
                try:
                    vector_store.add_texts(texts=all_chunks[start:end], metadatas=all_metadata[start:end])
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        time.sleep(2 * (attempt + 1)); continue
                    raise
            progress = 30 + int(((b + 1) / num_batches) * 70)
            job_update(job_id, progress=progress)
            print(f"[JOB {job_id}] batch {b+1}/{num_batches} ok ({end}/{num_chunks})")

        elapsed = time.time() - t0
        print(f"[JOB {job_id}] ✅ DONE em {elapsed:.1f}s")
        job_update(
            job_id, status="completed", stage="done", progress=100,
            result={"total_chunks": num_chunks, "total_files": len(files_data), "elapsed_seconds": round(elapsed, 1)},
        )
    except Exception as e:
        print(f"Erro Crítico no Job {job_id}:")
        print(traceback.format_exc())
        job_update(job_id, status="error", error=str(e))


# -------------------------------
# Helpers de inspeção do projeto (para o agente)
# -------------------------------
def list_project_sources(project_id: str) -> Dict[str, Any]:
    """Lista arquivos/fontes e registros disponíveis (visão geral para o agente decidir buscas)."""
    try:
        vs = get_vector_store(project_id)
        # cobertura aproximada: amostra de docs via similarity ampla
        sample = vs.max_marginal_relevance_search("0000 0150 0200 M100 M200 C100", k=60, fetch_k=200)
    except Exception as e:
        return {"error": str(e), "sources": [], "registros": {}}
    sources = Counter()
    regs = Counter()
    kinds = Counter()
    for d in sample:
        sources[d.metadata.get("source", "?")] += 1
        regs[d.metadata.get("registro") or primary_registro(d.page_content) or "?"] += 1
        kinds[d.metadata.get("source_kind", "?")] += 1
    return {
        "sources": [{"name": k, "chunks_sample": v} for k, v in sources.most_common(20)],
        "registros": dict(regs.most_common(30)),
        "source_kinds": dict(kinds),
        "note": "Amostragem aproximada (top 60 chunks). Use search_sped/search_driva para buscas dirigidas.",
    }


# -------------------------------
# AGENTIC TOOLS — execução real das buscas
# -------------------------------
def _doc_to_payload(d: Any, max_chars: int = AGENTIC_MAX_CHUNK_CHARS) -> Dict[str, Any]:
    content = (d.page_content or "")[:max_chars]
    reg = d.metadata.get("registro") or primary_registro(d.page_content) or "?"
    return {
        "source": d.metadata.get("source", "?"),
        "chunk_index": d.metadata.get("chunk_index"),
        "registro": reg,
        "source_kind": d.metadata.get("source_kind", "?"),
        "text": content,
    }


def tool_search_sped(project_id: str, query: str, k: int = AGENTIC_DEFAULT_K, registro: Optional[str] = None) -> Dict[str, Any]:
    k = max(1, min(int(k or AGENTIC_DEFAULT_K), AGENTIC_MAX_K))
    vs = get_vector_store(project_id)
    flt: Dict[str, Any] = {"source_kind": "sped"}
    if registro:
        flt["registro"] = registro
    try:
        docs = vs.max_marginal_relevance_search(query, k=k, fetch_k=k * 3, filter=flt)
    except Exception as e:
        # fallback sem filtro
        try:
            docs = vs.max_marginal_relevance_search(query, k=k, fetch_k=k * 3)
        except Exception as e2:
            return {"error": f"search_sped falhou: {e2}", "results": []}
    # aplica cap 0450
    docs, dropped = cap_registro_chunks(docs, "0450", MAX_0450_CHUNKS)
    return {
        "query": query,
        "k": k,
        "registro_filter": registro,
        "dropped_0450": dropped,
        "results": [_doc_to_payload(d) for d in docs],
    }


def tool_search_driva(project_id: str, query: str, k: int = 5) -> Dict[str, Any]:
    k = max(1, min(int(k or 5), 10))
    vs = get_vector_store(project_id)
    try:
        docs = vs.max_marginal_relevance_search(query, k=k, fetch_k=k * 3, filter=_filter_driva())
    except Exception as e:
        return {"error": f"search_driva falhou: {e}", "results": []}
    return {"query": query, "k": k, "results": [_doc_to_payload(d) for d in docs]}


def tool_get_periodo(project_id: str) -> Dict[str, Any]:
    vs = get_vector_store(project_id)
    try:
        docs = vs.max_marginal_relevance_search(
            "0000 abertura período DT_INI DT_FIN CNPJ", k=6, fetch_k=20, filter=_filter_sped()
        )
    except Exception:
        docs = vs.max_marginal_relevance_search("0000 abertura período DT_INI DT_FIN CNPJ", k=6, fetch_k=20)
    periodo = extract_periodo_from_docs(docs)
    return {"periodo": periodo}


def tool_list_sources(project_id: str) -> Dict[str, Any]:
    return list_project_sources(project_id)


# Schema das tools para o OpenAI tool-calling
AGENTIC_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_sources",
            "description": (
                "Lista arquivos SPED disponíveis no projeto e os registros (ex: M100, C170, 0000) "
                "mais frequentes. Use no início para entender o que existe antes de buscar."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_periodo",
            "description": "Retorna o período fiscal analisado (DT_INI / DT_FIN do registro 0000).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_sped",
            "description": (
                "Busca semântica em chunks SPED do projeto. Use perguntas específicas (ex: "
                "'créditos PIS sobre devoluções M100', 'apuração ICMS E110', 'itens nota fiscal C170 CFOP'). "
                "Filtre por registro quando souber (ex: 'M200', 'C100')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Pergunta ou termos de busca."},
                    "k": {"type": "integer", "description": f"Nº de chunks (1..{AGENTIC_MAX_K}). Default {AGENTIC_DEFAULT_K}."},
                    "registro": {"type": "string", "description": "Registro SPED para filtrar (ex: M100, C170, E110)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_driva",
            "description": (
                "Busca contexto de negócio (Driva): porte, CNAE, regime tributário, sócios, situação cadastral. "
                "NUNCA use como evidência numérica/fiscal — apenas para qualificar a empresa."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "description": "1..10. Default 5."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Encerre o loop e produza o SUMÁRIO FINAL completo em Markdown, obedecendo o template. "
                "Chame esta tool quando tiver evidências suficientes (ou quando ficar claro que não há dados)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "markdown": {
                        "type": "string",
                        "description": "Sumário final em Markdown, com todas as seções do template preenchidas.",
                    }
                },
                "required": ["markdown"],
            },
        },
    },
]


# -------------------------------
# POS-PROCESSAMENTO (parse markdown -> structured)
# -------------------------------
RE_CALC_LINE = re.compile(r"([A-Z][^=:]{3,60})\s*[:=]\s*(R\$\s?[\d\.,]+|\d[\d\.,]*)\s*(.*)")
RE_EVIDENCE_PIPE = re.compile(r"\|\s*([0-9A-Z]{4})\s*\|[^|\n]{2,200}\|")
RE_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[\.\)])\s+(.{15,})$", re.MULTILINE)


def _split_markdown_sections(markdown: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    if not markdown:
        return sections
    lines = markdown.splitlines()
    current_title = "_intro"
    buf: List[str] = []
    for ln in lines:
        m = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", ln)
        if m:
            sections[current_title] = "\n".join(buf).strip()
            current_title = re.sub(r"[^a-z0-9]+", "_", m.group(1).lower()).strip("_")
            buf = []
        else:
            buf.append(ln)
    sections[current_title] = "\n".join(buf).strip()
    return sections


def parse_summary_markdown(markdown: str) -> Dict[str, List[Any]]:
    result = {"insights": [], "calculations": [], "data_crossings": [], "source_references": []}
    if not markdown:
        return result
    sections = _split_markdown_sections(markdown)

    seen_insights = set()
    for title, body in sections.items():
        if not any(kw in title for kw in ("sumar", "exec", "recomend", "oportun", "intelig", "achad", "risco")):
            continue
        for m in RE_BULLET.finditer(body):
            text = m.group(1).strip()
            text = re.sub(r"\*\*", "", text)
            if "dado não disponível" in text.lower():
                continue
            if text.lower() in seen_insights:
                continue
            seen_insights.add(text.lower())
            if len(result["insights"]) < 40:
                result["insights"].append(text)

    seen_calcs = set()
    for m in RE_CALC_LINE.finditer(markdown):
        desc = m.group(1).strip(" -*").strip()
        val = m.group(2).strip()
        if "dado não disponível" in desc.lower() or "dado não disponível" in val.lower():
            continue
        key = f"{desc}|{val}"
        if key in seen_calcs:
            continue
        seen_calcs.add(key)
        result["calculations"].append({"description": desc, "value": val, "formula": ""})
        if len(result["calculations"]) >= 30:
            break

    seen_refs = set()
    for m in RE_EVIDENCE_PIPE.finditer(markdown):
        full = m.group(0).strip()
        reg = m.group(1).strip()
        if full in seen_refs:
            continue
        seen_refs.add(full)
        result["source_references"].append({"registro": reg, "excerpt": full[:400], "relevance": "evidência SPED"})
        if len(result["source_references"]) >= 40:
            break

    paragraphs = re.split(r"\n\s*\n", markdown)
    seen_cross = set()
    for p in paragraphs:
        regs = set(re.findall(r"\b([A-Z]\d{3})\b", p))
        regs |= set(re.findall(r"\|([0-9A-Z]{4})\|", p))
        if len(regs) >= 2 and len(p) < 1500:
            short = p.strip().replace("\n", " ")
            short_key = short[:120]
            if short_key in seen_cross:
                continue
            seen_cross.add(short_key)
            result["data_crossings"].append({"description": short[:600], "sources": sorted(regs), "result": ""})
            if len(result["data_crossings"]) >= 20:
                break
    return result


# -------------------------------
# AGENTIC SYSTEM PROMPT
# -------------------------------
AGENTIC_SYSTEM_PROMPT = """\
Você é um AGENTE auditor fiscal especialista em SPED (EFD PIS/COFINS, ICMS/IPI).
Sua missão: produzir um sumário fiscal de altíssima qualidade obedecendo ESTRITAMENTE o template do usuário.

Você NÃO recebe os documentos de antemão. Você precisa BUSCÁ-LOS sob demanda usando as ferramentas (tools):
- list_sources: visão geral dos arquivos e registros disponíveis no projeto.
- get_periodo: período fiscal (DT_INI/DT_FIN do 0000).
- search_sped(query, k, registro?): busca semântica nos chunks SPED. Use perguntas DIFERENTES para cobrir cada seção do template.
- search_driva(query, k): contexto de negócio (porte, CNAE, regime, sócios). NUNCA como evidência numérica.
- finish(markdown): encerra e entrega o sumário final em Markdown.

ESTRATÉGIA OBRIGATÓRIA:
1. Comece com list_sources e get_periodo para entender o que existe.
2. Para CADA seção/oportunidade do template, faça pelo menos UMA busca dirigida em search_sped.
   - Ex: "M200 apuração contribuição PIS COFINS valor devido", "C170 itens NCM CST CFOP", "E110 apuração ICMS".
   - Se a primeira busca não trouxer evidência, REFINE a query (sinônimos, código do registro) antes de desistir.
3. Faça no máximo {max_searches} buscas. Não repita queries idênticas.
4. Se NÃO encontrar evidência real para um item:
   - NÃO invente números.
   - NÃO escreva "Dado não disponível" repetidamente — OMITA a linha/coluna/oportunidade,
     ou agrupe num bloco curto "Itens sem evidência nos arquivos analisados".
5. Toda afirmação numérica/fiscal DEVE citar a fonte SPED (arquivo + registro + trecho literal).
6. Driva pode contextualizar porte/CNAE/regime, mas NUNCA gerar cálculo.
7. Registro 0450 é informação complementar — não use como base de impacto/ROI.
8. Substitua qualquer placeholder "X.N" pelo número real do capítulo.
9. Sempre que mencionar período, use exatamente o retornado por get_periodo.
10. Quando tiver evidências suficientes (ou ficar claro que não há mais o que buscar), chame finish(markdown=...).

FORMATO DO SUMÁRIO FINAL:
- Markdown bem estruturado, com títulos hierárquicos.
- Tabelas só quando houver dados reais para preencher todas as colunas relevantes.
- Cite trechos SPED entre aspas ou em blockquote, indicando arquivo e registro.
"""


def _run_agentic_summary(req: SummaryRequest, job_id: str) -> Dict[str, Any]:
    """
    Loop agêntico:
      - Modelo recebe template + tools.
      - Faz buscas iterativas no Chroma via tool calls.
      - Encerra com finish(markdown).
    """
    project_id = req.project_id
    template = req.template

    # Mensagem inicial
    system_msg = AGENTIC_SYSTEM_PROMPT.format(max_searches=AGENTIC_MAX_SEARCHES)
    user_msg = f"""\
TEMPLATE DO USUÁRIO (instruções de conteúdo e formato do sumário):
\"\"\"
{template}
\"\"\"

OBJETIVO INICIAL: {req.query or "gerar sumário geral"}
PROJECT_ID: {project_id}

Inicie pela exploração (list_sources / get_periodo), depois faça buscas dirigidas para CADA seção do template.
Quando tiver evidências suficientes, chame finish(markdown=...).
"""

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    total_searches = 0
    total_tool_calls = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    final_markdown: Optional[str] = None
    trace: List[Dict[str, Any]] = []

    for step in range(AGENTIC_MAX_ITERATIONS):
        job_update(job_id, stage=f"agent_step_{step+1}")
        try:
            resp = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=AGENTIC_TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.1,
            )
        except Exception as e:
            print(f"[AGENT][{job_id}] erro na chamada LLM step={step}: {e}")
            raise

        if resp.usage:
            total_prompt_tokens += resp.usage.prompt_tokens or 0
            total_completion_tokens += resp.usage.completion_tokens or 0

        msg = resp.choices[0].message
        # Anexa mensagem do assistente ao histórico
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in (msg.tool_calls or [])
            ] if msg.tool_calls else None,
        })

        if not msg.tool_calls:
            # Modelo não chamou tool — pode ter respondido em texto.
            # Se tiver conteúdo razoável, aceita como final.
            if msg.content and len(msg.content.strip()) > 200:
                final_markdown = msg.content
                trace.append({"step": step, "action": "assistant_text_final"})
                break
            # Caso contrário, força encerramento
            messages.append({
                "role": "user",
                "content": "Você não chamou nenhuma tool. Se já tem dados suficientes, chame finish(markdown=...). Caso contrário, faça mais buscas com search_sped.",
            })
            continue

        # Executa cada tool call
        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            total_tool_calls += 1
            print(f"[AGENT][{job_id}] step={step+1} tool={name} args={str(args)[:200]}")

            tool_result: Any
            if name == "finish":
                final_markdown = args.get("markdown") or ""
                trace.append({"step": step, "action": "finish", "len": len(final_markdown)})
                # Adiciona resposta da tool (obrigatório pelo protocolo)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"ok": True}),
                })
                break

            if name == "list_sources":
                tool_result = tool_list_sources(project_id)
            elif name == "get_periodo":
                tool_result = tool_get_periodo(project_id)
            elif name == "search_sped":
                if total_searches >= AGENTIC_MAX_SEARCHES:
                    tool_result = {"error": f"Limite de buscas atingido ({AGENTIC_MAX_SEARCHES}). Chame finish."}
                else:
                    total_searches += 1
                    tool_result = tool_search_sped(
                        project_id,
                        query=args.get("query", ""),
                        k=args.get("k", AGENTIC_DEFAULT_K),
                        registro=args.get("registro"),
                    )
            elif name == "search_driva":
                tool_result = tool_search_driva(
                    project_id, query=args.get("query", ""), k=args.get("k", 5),
                )
            else:
                tool_result = {"error": f"tool desconhecida: {name}"}

            trace.append({"step": step, "action": name, "args": args, "n_results": len(tool_result.get("results", [])) if isinstance(tool_result, dict) else None})

            # Trunca payload se muito grande para não estourar contexto
            payload_str = json.dumps(tool_result, ensure_ascii=False)
            if len(payload_str) > 60000:
                payload_str = payload_str[:60000] + ' ...[TRUNCATED]"}'
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": payload_str,
            })

        if final_markdown is not None:
            break
    else:
        print(f"[AGENT][{job_id}] ⚠️ atingiu MAX_ITERATIONS={AGENTIC_MAX_ITERATIONS} sem finish")

    if not final_markdown:
        # Último recurso: pedir um fechamento explícito
        messages.append({
            "role": "user",
            "content": "Encerre AGORA chamando finish(markdown=...) com o melhor sumário possível dado o que já buscou.",
        })
        try:
            resp = openai_client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=AGENTIC_TOOLS_SCHEMA,
                tool_choice={"type": "function", "function": {"name": "finish"}},
                temperature=0.1,
            )
            if resp.usage:
                total_prompt_tokens += resp.usage.prompt_tokens or 0
                total_completion_tokens += resp.usage.completion_tokens or 0
            tcs = resp.choices[0].message.tool_calls or []
            if tcs:
                args = json.loads(tcs[0].function.arguments or "{}")
                final_markdown = args.get("markdown") or ""
        except Exception as e:
            print(f"[AGENT][{job_id}] fallback finish falhou: {e}")

    if not final_markdown:
        raise Exception("Agente não produziu sumário final.")

    return {
        "markdown": final_markdown,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "searches": total_searches,
        "tool_calls": total_tool_calls,
        "iterations": min(step + 1, AGENTIC_MAX_ITERATIONS),
        "trace": trace,
    }


# -------------------------------
# PROCESS SUMMARY JOB (agentic-first, com fallback one-shot)
# -------------------------------
def process_summary_job(job_id: str, req: SummaryRequest):
    t_summary_start = time.time()
    try:
        job_update(job_id, status="processing", stage="starting")
        print(f"[SUMMARY][{job_id}] 🚀 START | agentic={req.agentic}")

        if req.enrichment:
            print(f"[SUMMARY][{job_id}] ⚠️ enrichment no body está DEPRECATED — ignorando. Use /enrichment para indexar.")

        # ---- MODO AGÊNTICO (default) ----
        if req.agentic:
            try:
                job_update(job_id, stage="agentic_rag")
                agent_out = _run_agentic_summary(req, job_id)
                content_str = agent_out["markdown"]
                structured = parse_summary_markdown(content_str)
                periodo = tool_get_periodo(req.project_id).get("periodo")
                generation_time_ms = int((time.time() - t_summary_start) * 1000)

                job_update(
                    job_id, status="completed", stage="done",
                    result={
                        "mode": "agentic",
                        "content": content_str,
                        "model": LLM_MODEL, "model_used": LLM_MODEL,
                        "tokens_used": agent_out["total_tokens"],
                        "prompt_tokens": agent_out["prompt_tokens"],
                        "completion_tokens": agent_out["completion_tokens"],
                        "generation_time_ms": generation_time_ms,
                        "periodo_detectado": periodo,
                        "insights": structured["insights"],
                        "calculations": structured["calculations"],
                        "data_crossings": structured["data_crossings"],
                        "source_references": structured["source_references"],
                        "agent": {
                            "iterations": agent_out["iterations"],
                            "searches": agent_out["searches"],
                            "tool_calls": agent_out["tool_calls"],
                            "trace": agent_out["trace"],
                        },
                    },
                )
                print(f"[SUMMARY][{job_id}] ✅ AGENTIC DONE | iters={agent_out['iterations']} | searches={agent_out['searches']} | tokens={agent_out['total_tokens']} | {generation_time_ms}ms")
                return
            except Exception as e:
                print(f"[SUMMARY][{job_id}] ⚠️ agentic falhou ({e}); caindo para one-shot legado")
                print(traceback.format_exc())
                # Continua para modo legado abaixo

        # ---- MODO LEGADO (one-shot RAG) — fallback / req.agentic=False ----
        mode = "strategic" if "DOCUMENTO 1" in req.template else "audit"
        print(f"[SUMMARY][{job_id}] Mode legado: {mode}")

        job_update(job_id, stage="retrieving_context")
        effective_k = req.k or 10
        context = get_context(req.query, req.project_id, effective_k)
        context = context[:12000]

        try:
            vs = get_vector_store(req.project_id)
            periodo_docs = vs.max_marginal_relevance_search(
                "0000 abertura período DT_INI DT_FIN CNPJ", k=6, fetch_k=20,
            )
            periodo = extract_periodo_from_docs(periodo_docs)
        except Exception:
            periodo = None

        periodo_block = f"\nPERÍODO ANALISADO: {periodo}\n" if periodo else ""
        prompt = f"""Você é um auditor fiscal especialista em SPED.

{periodo_block}
CONTEXTO:
{context}

INSTRUÇÕES (template):
{req.template}

REGRAS:
- NÃO invente números. Cite arquivo e registro de cada evidência.
- Se não houver evidência, OMITA o item (não escreva "Dado não disponível" repetidamente).
- Driva = apenas contexto de negócio, nunca evidência numérica.
"""
        with get_openai_callback() as cb:
            response = llm.invoke(prompt)
            tokens_used = cb.total_tokens
            ptok = cb.prompt_tokens
            ctok = cb.completion_tokens

        content_str = response.content
        structured = parse_summary_markdown(content_str)
        generation_time_ms = int((time.time() - t_summary_start) * 1000)

        job_update(
            job_id, status="completed", stage="done",
            result={
                "mode": "legacy_oneshot",
                "content": content_str,
                "model": LLM_MODEL, "model_used": LLM_MODEL,
                "tokens_used": tokens_used,
                "prompt_tokens": ptok,
                "completion_tokens": ctok,
                "generation_time_ms": generation_time_ms,
                "periodo_detectado": periodo,
                "insights": structured["insights"],
                "calculations": structured["calculations"],
                "data_crossings": structured["data_crossings"],
                "source_references": structured["source_references"],
            },
        )
        print(f"[SUMMARY][{job_id}] ✅ LEGACY DONE | tokens={tokens_used} | {generation_time_ms}ms")

    except Exception as e:
        print(f"[SUMMARY][{job_id}] ❌ ERROR")
        print(traceback.format_exc())
        job_update(job_id, status="error", error=str(e))


# -------------------------------
# ENRICHMENT (Driva)
# -------------------------------
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


@app.post("/enrichment")
async def upload_enrichment(req: EnrichmentRequest):
    try:
        vector_store = get_vector_store(req.project_id)
        filtered = preprocess_driva(req.enrichment)
        if not filtered:
            return {"status": "success", "chunks_saved": 0, "message": "Nenhum campo Driva relevante encontrado"}

        enrichment_text = enrichment_to_text(filtered)
        chunks = text_splitter.split_text(enrichment_text)
        texts, metadatas = [], []
        for idx, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({
                "project_id": req.project_id,
                "type": "enrichment",
                "source_kind": "driva",
                "source": req.source,
                "chunk_index": idx,
                "registro": "enrichment",
            })
        vector_store.add_texts(texts=texts, metadatas=metadatas)
        return {
            "status": "success",
            "chunks_saved": len(chunks),
            "fields_kept": list(filtered.keys()) if isinstance(filtered, dict) else [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# UPLOAD (legado)
# -------------------------------
@app.post("/upload")
async def upload_documents(project_id: str, files: List[UploadFile] = File(...)):
    try:
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id obrigatório")
        job_id = str(uuid.uuid4())

        async def process_single_file(file: UploadFile):
            content = await file.read()
            text = smart_decode_sped(content)
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


# -------------------------------
# /process-paths
# -------------------------------
def _download_file(path: str) -> Optional[dict]:
    try:
        data = supabase.storage.from_(BUCKET_NAME).download(path)
        text = smart_decode_sped(data)
        if not text.strip():
            return None
        filename = path.split("/")[-1]
        return {"filename": filename, "text": text}
    except Exception as e:
        print(f"[download] falhou {path}: {e}")
        return None


@app.post("/process-paths")
def process_paths(req: ProcessPathsRequest):
    try:
        if not req.project_id or not req.paths:
            raise HTTPException(status_code=400, detail="project_id e paths obrigatórios")

        job_id = str(uuid.uuid4())
        job_create(job_id, kind="process", project_id=req.project_id)

        def runner():
            try:
                job_update(job_id, status="processing", stage="downloading", progress=1)
                t0 = time.time()
                with ThreadPoolExecutor(max_workers=16) as ex:
                    results = list(ex.map(_download_file, req.paths))
                files_data = [r for r in results if r is not None]
                print(f"[JOB {job_id}] download {len(files_data)}/{len(req.paths)} em {time.time()-t0:.1f}s")
                if not files_data:
                    job_update(job_id, status="error", error="Nenhum arquivo válido baixado")
                    return
                process_job(job_id, files_data, req.project_id)
            except Exception as e:
                print(traceback.format_exc())
                job_update(job_id, status="error", error=str(e))

        threading.Thread(target=runner).start()
        return {"job_id": job_id, "project_id": req.project_id, "total_files": len(req.paths)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------
# DELETE PROJECT
# -------------------------------
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


# -------------------------------
# STATUS
# -------------------------------
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


# -------------------------------
# SUMMARY
# -------------------------------
@app.post("/generate-summary")
async def generate_summary(req: SummaryRequest):
    try:
        job_id = str(uuid.uuid4())
        job_create(job_id, kind="summary", project_id=req.project_id)
        threading.Thread(target=process_summary_job, args=(job_id, req)).start()
        return {"job_id": job_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
