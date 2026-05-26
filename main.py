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

# Limite seguro de caracteres por chunk (~8191 tokens * 4 chars/token = ~32k; usamos margem)
MAX_CHARS_PER_CHUNK = 24000
# OpenAI rejeita string vazia; usamos espaço para manter alinhamento
EMPTY_PLACEHOLDER = " "


def _sanitize_texts(texts: List[str]) -> List[str]:
    """Garante que toda entrada tenha pelo menos 1 char e não exceda o limite do modelo."""
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
# 🆕 #3 — DECODER INTELIGENTE PARA SPED (BOM / latin-1 / cp1252)
# -------------------------------
def smart_decode_sped(raw: bytes) -> str:
    """
    Detecta encoding correto de arquivos SPED preservando acentuação.
    Ordem: BOM UTF-8/UTF-16 -> utf-8 estrito -> latin-1 -> cp1252.
    Nunca usa errors='ignore' (que era o que perdia acentos).
    """
    if not raw:
        return ""

    # 1) BOM explícito
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")

    # 2) Tenta UTF-8 estrito (SPEDs mais novos)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass

    # 3) latin-1 / ISO-8859-1 (padrão histórico do SPED)
    try:
        decoded = raw.decode("latin-1")
        # Heurística: se houver muitos caracteres "estranhos" comuns em cp1252 mal-lido,
        # tenta cp1252 que cobre aspas tipográficas, travessões, etc.
        if any(c in decoded for c in ("\x80", "\x82", "\x83", "\x84", "\x85", "\x86", "\x87", "\x88", "\x89")):
            try:
                return raw.decode("cp1252")
            except UnicodeDecodeError:
                return decoded
        return decoded
    except UnicodeDecodeError:
        pass

    # 4) Último recurso
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
        res = supabase.table(JOBS_TABLE).select(
            "*").eq("id", job_id).limit(1).execute()
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
        return JSONResponse(status_code=401, content={
                            "detail": "Unauthorized"})
    return await call_next(request)


app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -------------------------------
# CONFIG
# -------------------------------
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150          # ⚡ menor overlap = menos chunks
PERSIST_DIR = "/data/chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY não encontrada")

# ⚡ Cliente OpenAI nativo p/ batch de embeddings (até 2048 inputs por request)
openai_client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 256

# 🆕 #2 — modelo declarado em const para fácil propagação ao Supabase
LLM_MODEL = "gpt-4.1"


class BatchOpenAIEmbeddings(Embeddings):
    """Embeddings em lote (256 textos por request) — ~10x mais rápido."""

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
                    resp = openai_client.embeddings.create(
                        model=EMBED_MODEL,
                        input=batch,
                    )
                    embeddings_batch = [d.embedding for d in resp.data]

                    if len(embeddings_batch) != len(batch):
                        raise RuntimeError(
                            f"OpenAI retornou {len(embeddings_batch)} embeddings "
                            f"para {len(batch)} inputs (batch {i}-{i+len(batch)}). "
                            f"Resposta desalinhada."
                        )
                    break

                except AuthenticationError as e:
                    logger.error("OpenAI auth error: %s", e)
                    raise RuntimeError(
                        "OpenAI API Key inválida ou revogada. "
                        "Verifique OPENAI_API_KEY no .env do backend."
                    ) from e

                except RateLimitError as e:
                    msg = str(e)
                    if "insufficient_quota" in msg or "exceeded your current quota" in msg:
                        logger.error("OpenAI sem créditos: %s", msg)
                        raise RuntimeError(
                            "OpenAI sem créditos disponíveis (insufficient_quota). "
                            "Adicione saldo em https://platform.openai.com/account/billing "
                            "e tente novamente."
                        ) from e

                    if attempt < 4:
                        wait = 2 ** attempt
                        logger.warning("Rate limit OpenAI (tentativa %d/5). Aguardando %ds...", attempt + 1, wait)
                        time.sleep(wait)
                        continue
                    raise

                except BadRequestError as e:
                    logger.error("OpenAI bad request no batch %d-%d: %s", i, i + len(batch), e)
                    raise

                except APIError as e:
                    if attempt < 4:
                        wait = 2 ** attempt
                        logger.warning("OpenAI APIError (tentativa %d/5): %s. Aguardando %ds...", attempt + 1, e, wait)
                        time.sleep(wait)
                        continue
                    raise

                except Exception as e:
                    msg = str(e)
                    if ("429" in msg or "rate" in msg.lower()) and attempt < 4:
                        time.sleep(2 ** attempt)
                        continue
                    raise

            if len(embeddings_batch) != len(batch):
                raise RuntimeError(
                    f"Falha ao gerar embeddings completos para batch {i}-{i+len(batch)} "
                    f"após todas as tentativas."
                )

            out.extend(embeddings_batch)

        if len(out) != len(texts):
            raise RuntimeError(
                f"Inconsistência crítica: {len(out)} embeddings gerados para {len(texts)} textos. "
                f"Isso causaria IndexError no Chroma."
            )

        return out

    def embed_query(self, text: str) -> List[float]:
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
        return resp.data[0].embedding


embeddings = BatchOpenAIEmbeddings()
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0, api_key=OPENAI_API_KEY)

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
# SPED helpers
# -------------------------------
SPED_REGISTRO_REGEX = re.compile(r"^\|([0-9A-Z]{4})\|", re.MULTILINE)

# 🆕 #5 — regex para extrair período do registro 0000 (DT_INI|DT_FIN são os campos 4 e 5)
# Formato típico: |0000|VER|COD_FIN|DT_INI|DT_FIN|NOME|CNPJ|...
SPED_0000_REGEX = re.compile(
    r"\|0000\|[^|]*\|[^|]*\|(\d{8})\|(\d{8})\|",
    re.MULTILINE,
)


def detect_registros(text: str) -> List[str]:
    if not text:
        return []
    found = SPED_REGISTRO_REGEX.findall(text)
    seen, out = set(), []
    for r in found:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def primary_registro(text: str) -> Optional[str]:
    if not text:
        return None
    matches = SPED_REGISTRO_REGEX.findall(text)
    if not matches:
        return None
    return Counter(matches).most_common(1)[0][0]


def extract_periodo_from_docs(docs) -> Optional[str]:
    """
    🆕 #5 — Varre chunks atrás de registros |0000| e extrai DT_INI/DT_FIN.
    Retorna string formatada tipo 'DD/MM/AAAA a DD/MM/AAAA' ou None.
    """
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
    # menor DT_INI, maior DT_FIN
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
# RAG HELPERS
# -------------------------------
def get_context(query: str, project_id: str, k: int = 10):
    try:
        vector_store = get_vector_store(project_id)
        docs = vector_store.max_marginal_relevance_search(query, k=k, fetch_k=k * 4)

        if not docs:
            return "Nenhum dado encontrado para este projeto."

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


def build_prompt(template: str, context: str, enrichment: dict | None, periodo: Optional[str] = None):
    enrichment_text = ""
    if enrichment:
        enrichment_text = f"""
====================
DADOS DE ENRIQUECIMENTO
====================
{enrichment}
"""

    # 🆕 #5 — bloco de período fixo no topo
    periodo_block = ""
    if periodo:
        periodo_block = f"""
====================
PERÍODO ANALISADO (extraído do registro 0000)
====================
{periodo}
"""

    return f"""
Você é um auditor fiscal especialista em SPED (EFD PIS/COFINS).

Sua função é identificar inconsistências reais, validar cálculos e garantir integridade dos dados.
{periodo_block}
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
8. NÃO use o registro 0450 como evidência principal de análise financeira.
   Priorize blocos com valores: M100, M200, M500, M600 (apuração) e C100, C170, C190, C500, D100.
9. Ao numerar seções, substitua qualquer placeholder `X.N` pelo número real do capítulo.
   NUNCA escreva `X.` literal no texto final (ex.: use "4.3" e não "X.3" ou ".3").
10. Se um item realmente não tiver dado, escreva "Dado não disponível nos arquivos analisados"
    SOMENTE uma vez por seção — não duplique.
11. Sempre que mencionar o período da análise, use exatamente o período informado no bloco
    "PERÍODO ANALISADO" acima. Só escreva "Dado não disponível" para período se aquele bloco
    estiver ausente.

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


class ProcessPathsRequest(BaseModel):
    project_id: str
    paths: List[str]


# -------------------------------
# PROCESS JOB (chunking + embedding em batch)
# -------------------------------
def process_job(job_id: str, files_data: List[dict], project_id: str):
    try:
        t0 = time.time()
        job_update(job_id, status="processing", stage="chunking", progress=5)

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
                    vector_store.add_texts(
                        texts=all_chunks[start:end],
                        metadatas=all_metadata[start:end],
                    )
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 2:
                        time.sleep(2 * (attempt + 1)); continue
                    raise
            progress = 30 + int(((b + 1) / num_batches) * 70)
            job_update(job_id, progress=progress)
            print(f"[JOB {job_id}] batch {b+1}/{num_batches} ok ({end}/{num_chunks})")

        elapsed = time.time() - t0
        print(f"[JOB {job_id}] ✅ DONE em {elapsed:.1f}s ({num_chunks} chunks, {len(files_data)} arquivos)")
        job_update(
            job_id, status="completed", stage="done", progress=100,
            result={"total_chunks": num_chunks, "total_files": len(files_data),
                    "elapsed_seconds": round(elapsed, 1)},
        )

    except Exception as e:
        print(f"Erro Crítico no Job {job_id}:")
        print(traceback.format_exc())
        job_update(job_id, status="error", error=str(e))


# -------------------------------
# 🆕 #1 — Multi-query retrieval ADAPTATIVO por template
# -------------------------------
# Queries direcionadas para registros de APURAÇÃO (prioridade máxima)
APURACAO_QUERIES = [
    "M100 M105 base de cálculo crédito PIS valor alíquota",
    "M200 M205 M210 apuração contribuição PIS COFINS valor devido",
    "M500 M505 M600 M605 M610 crédito COFINS débito apuração",
    "E110 E116 apuração ICMS débitos créditos saldo",
    "E520 E530 apuração IPI período",
]

# Queries para documentos fiscais com VALORES
FISCAIS_QUERIES = [
    "C100 nota fiscal valor total operação ICMS",
    "C170 item nota fiscal NCM CST CFOP valor unitário",
    "C190 C500 D100 análise consolidada itens",
]

# Queries para cadastros/contexto (peso menor)
CADASTRAIS_QUERIES = [
    "0000 abertura arquivo período empresa CNPJ",
    "0150 participantes fornecedores clientes",
    "0200 0220 itens produtos serviços fatores conversão",
    "totais consolidados ajustes créditos do período",
    "inconsistências divergências entre totais e itens valores zerados",
]


def _is_analise_completa(template: str) -> bool:
    """Detecta heurísticamente o template 'Análise Completa' / strategic."""
    if not template:
        return False
    t = template.lower()
    keywords = ["análise completa", "analise completa", "relatório executivo estratégico",
                "documento 1", "consolidação final", "consolidacao final"]
    return any(kw in t for kw in keywords)


def multi_query_retrieval(
    vector_store,
    base_query: str,
    per_query_k: int = 8,
    template: str = "",
) -> List[Any]:
    """
    Multi-step retrieval. Para Análise Completa:
      passo 1: APURACAO (M100/M200/M500/E110) com k maior
      passo 2: FISCAIS  (C100/C170/C190)
      passo 3: CADASTRAIS (0000/0150) com k menor
    Para audit comum: mantém comportamento anterior (queries unificadas).
    """
    analise_completa = _is_analise_completa(template)

    if analise_completa:
        # Multi-step ponderado
        steps: List[Tuple[List[str], int, str]] = [
            (APURACAO_QUERIES, max(per_query_k, 10), "APURACAO"),
            (FISCAIS_QUERIES, max(per_query_k, 8), "FISCAIS"),
            (CADASTRAIS_QUERIES, max(per_query_k // 2, 4), "CADASTRAIS"),
        ]
        # Inclui base_query no primeiro passo
        steps[0] = ([base_query] + steps[0][0], steps[0][1], steps[0][2])
    else:
        # Comportamento original consolidado
        legacy_queries = [base_query] + APURACAO_QUERIES[:2] + FISCAIS_QUERIES[:2] + CADASTRAIS_QUERIES[3:]
        steps = [(legacy_queries, per_query_k, "LEGACY")]

    seen, merged = set(), []
    for queries, k_step, label in steps:
        before = len(merged)
        for q in queries:
            try:
                docs = vector_store.max_marginal_relevance_search(
                    q, k=k_step, fetch_k=k_step * 3,
                    filter={"type": "document"},
                )
            except Exception as e:
                print(f"[multi_query_retrieval][{label}] falhou query={q[:40]!r}: {e}")
                continue
            for d in docs:
                key = (d.metadata.get("source"), d.metadata.get("chunk_index"))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(d)
        print(f"[multi_query_retrieval][{label}] +{len(merged)-before} novos (total={len(merged)})")

    return merged


def log_registro_distribution(job_id: str, docs, label: str):
    dist = Counter()
    for d in docs:
        reg = d.metadata.get("registro") or primary_registro(d.page_content) or "unknown"
        dist[reg] += 1
    pretty = ", ".join(f"{k}={v}" for k, v in dist.most_common())
    print(f"[SUMMARY][{job_id}] DIST {label}: total={len(docs)} | {pretty}")
    return dist


# -------------------------------
# 🆕 #4 — POS-PROCESSAMENTO ESTRUTURAL DO MARKDOWN
# -------------------------------
RE_CALC_LINE = re.compile(r"([A-Z][^=:]{3,60})\s*[:=]\s*(R\$\s?[\d\.,]+|\d[\d\.,]*)\s*(.*)")
RE_EVIDENCE_PIPE = re.compile(r"\|\s*([0-9A-Z]{4})\s*\|[^|\n]{2,200}\|")
RE_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[\.\)])\s+(.{15,})$", re.MULTILINE)


def _split_markdown_sections(markdown: str) -> Dict[str, str]:
    """Divide markdown por headers ##/### retornando dict {titulo_normalizado: corpo}."""
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
    """
    Faz best-effort para extrair:
      - insights: bullets das seções de sumário/recomendações/inteligência
      - calculations: linhas com 'Nome: R$ valor' ou 'X = N'
      - data_crossings: blocos que citam múltiplos registros (ex.: C100 vs C170)
      - source_references: trechos no formato |REG|...| extraídos como evidências
    """
    result = {
        "insights": [],
        "calculations": [],
        "data_crossings": [],
        "source_references": [],
    }
    if not markdown:
        return result

    sections = _split_markdown_sections(markdown)

    # --- insights: bullets das seções relevantes ---
    insight_keywords = ("sumario", "sumário", "executivo", "recomenda", "oportunidade",
                        "inteligencia", "inteligência", "achados", "risco")
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

    # --- calculations: linhas tipo "Algo: R$ 1.234,56" ---
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

    # --- source_references: trechos pipe-delimitados ---
    seen_refs = set()
    for m in RE_EVIDENCE_PIPE.finditer(markdown):
        full = m.group(0).strip()
        reg = m.group(1).strip()
        if full in seen_refs:
            continue
        seen_refs.add(full)
        result["source_references"].append({
            "registro": reg,
            "excerpt": full[:400],
            "relevance": "evidência SPED",
        })
        if len(result["source_references"]) >= 40:
            break

    # --- data_crossings: parágrafos que mencionam 2+ registros SPED distintos ---
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
            result["data_crossings"].append({
                "description": short[:600],
                "sources": sorted(regs),
                "result": "",
            })
            if len(result["data_crossings"]) >= 20:
                break

    return result


# -------------------------------
# PROCESS SUMMARY JOB
# -------------------------------
def process_summary_job(job_id: str, req: SummaryRequest):
    t_summary_start = time.time()
    try:
        job_update(job_id, status="processing", stage="starting")
        print(f"[SUMMARY][{job_id}] 🚀 START")

        mode = "strategic" if "DOCUMENTO 1" in req.template else "audit"
        is_completa = _is_analise_completa(req.template)
        print(f"[SUMMARY][{job_id}] Mode: {mode} | AnaliseCompleta: {is_completa}")

        total_tokens_used = 0
        prompt_tokens = 0
        completion_tokens = 0

        if mode == "audit":
            job_update(job_id, stage="retrieving_context")
            # 🆕 #1 — k adaptativo: 30 para análise completa, k da req para resto
            effective_k = 30 if is_completa else (req.k or 10)
            context = get_context(req.query, req.project_id, effective_k)
            print(f"[SUMMARY][{job_id}] Context size: {len(context)} (k={effective_k})")
            context = context[:14000 if is_completa else 12000]

            # 🆕 #5 — extrai período direto do vector store p/ injetar no prompt
            try:
                vs = get_vector_store(req.project_id)
                periodo_docs = vs.max_marginal_relevance_search(
                    "0000 abertura período DT_INI DT_FIN CNPJ", k=6, fetch_k=20
                )
                periodo = extract_periodo_from_docs(periodo_docs)
                print(f"[SUMMARY][{job_id}] Período extraído: {periodo}")
            except Exception as e:
                print(f"[SUMMARY][{job_id}] ⚠️ falha extraindo período: {e}")
                periodo = None

            job_update(job_id, stage="building_prompt")
            prompt = build_prompt(req.template, context, req.enrichment, periodo)
            print(f"[SUMMARY][{job_id}] Prompt size: {len(prompt)}")

            job_update(job_id, stage="llm_call")
            with get_openai_callback() as cb:
                response = llm.invoke(prompt)
                total_tokens_used = cb.total_tokens
                prompt_tokens = cb.prompt_tokens
                completion_tokens = cb.completion_tokens

            generation_time_ms = int((time.time() - t_summary_start) * 1000)
            content_str = response.content
            structured = parse_summary_markdown(content_str)

            job_update(
                job_id, status="completed", stage="done",
                result={
                    "mode": mode,
                    "content": content_str,
                    # 🆕 #2 — metadados de geração
                    "model": LLM_MODEL,
                    "model_used": LLM_MODEL,
                    "tokens_used": total_tokens_used,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "generation_time_ms": generation_time_ms,
                    "periodo_detectado": periodo,
                    # 🆕 #4 — campos estruturados extraídos
                    "insights": structured["insights"],
                    "calculations": structured["calculations"],
                    "data_crossings": structured["data_crossings"],
                    "source_references": structured["source_references"],
                },
            )

        else:
            job_update(job_id, stage="multi_step_rag")
            vector_store = get_vector_store(req.project_id)

            # 🆕 #1 — per_query_k adaptativo
            per_q_k = 12 if is_completa else 8
            docs = multi_query_retrieval(
                vector_store, req.query,
                per_query_k=per_q_k,
                template=req.template,
            )
            print(f"[SUMMARY][{job_id}] Docs após multi-query: {len(docs)} (per_q_k={per_q_k})")

            if not docs:
                print(f"[SUMMARY][{job_id}] ⚠️ multi-query vazio, fallback simples")
                docs = vector_store.max_marginal_relevance_search(
                    req.query, k=30 if is_completa else 20, fetch_k=60
                )

            log_registro_distribution(job_id, docs, "pré-cap")
            docs, dropped = cap_registro_chunks(docs, "0450", MAX_0450_CHUNKS)
            print(f"[SUMMARY][{job_id}] 0450 descartados: {dropped}")
            log_registro_distribution(job_id, docs, "pós-cap")

            # 🆕 #5 — extrai período dos próprios docs recuperados (ou via query dedicada)
            periodo = extract_periodo_from_docs(docs)
            if not periodo:
                try:
                    extra = vector_store.max_marginal_relevance_search(
                        "0000 abertura período DT_INI DT_FIN", k=6, fetch_k=20
                    )
                    periodo = extract_periodo_from_docs(extra)
                except Exception:
                    periodo = None
            print(f"[SUMMARY][{job_id}] Período extraído: {periodo}")

            for i, doc in enumerate(docs, start=1):
                src = doc.metadata.get("source", "?")
                reg = doc.metadata.get("registro") or primary_registro(doc.page_content) or "?"
                print(f"[SUMMARY][{job_id}] Doc {i}/{len(docs)} reg={reg} source={src}")

            partial_results = []
            with get_openai_callback() as cb_extract:
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

                total_tokens_used += cb_extract.total_tokens
                prompt_tokens += cb_extract.prompt_tokens
                completion_tokens += cb_extract.completion_tokens

            print(f"[SUMMARY][{job_id}] válidos: {len(partial_results)} | tokens extração={cb_extract.total_tokens}")
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

            # 🆕 #1 — agregado maior para análise completa
            agg_limit = 26000 if is_completa else 20000
            aggregated = "\n\n".join(filtered_results)[:agg_limit]
            print(f"[SUMMARY][{job_id}] Aggregated size: {len(aggregated)} (limit={agg_limit})")

            enrichment_block = ""
            if req.enrichment:
                enrichment_block = f"""
            ====================
            DADOS DE ENRIQUECIMENTO (CADASTRAL — COMPLEMENTAR)
            ====================
            {req.enrichment}
            """

            # 🆕 #5 — bloco de período fixo
            periodo_block = ""
            if periodo:
                periodo_block = f"""
            ====================
            PERÍODO ANALISADO (extraído do registro 0000)
            ====================
            {periodo}
            """

            final_prompt = f"""
            {req.template}
            {periodo_block}
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
            6. NÃO duplique títulos de seção.
            7. Registros 0450 são INFORMAÇÕES COMPLEMENTARES. NÃO os use como base para
               cálculo de impacto financeiro ou ROI — esses cálculos devem vir de
               M100/M200/M500/M600 e itens C170. Se faltar essa base, declare ausência
               explicitamente em vez de citar 0450.
            8. Sempre que mencionar o período da análise, use exatamente o período informado
               no bloco "PERÍODO ANALISADO" acima. Só escreva "Dado não disponível" para
               período se aquele bloco estiver ausente.
            """

            job_update(job_id, stage="final_llm")
            with get_openai_callback() as cb_final:
                final = llm.invoke(final_prompt)
                total_tokens_used += cb_final.total_tokens
                prompt_tokens += cb_final.prompt_tokens
                completion_tokens += cb_final.completion_tokens

            generation_time_ms = int((time.time() - t_summary_start) * 1000)
            content_str = final.content

            # 🆕 #4 — parse estrutural
            structured = parse_summary_markdown(content_str)

            job_update(
                job_id, status="completed", stage="done",
                result={
                    "mode": mode,
                    "content": content_str,
                    # 🆕 #2 — metadados de geração (top-level)
                    "model": LLM_MODEL,
                    "model_used": LLM_MODEL,
                    "tokens_used": total_tokens_used,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "generation_time_ms": generation_time_ms,
                    "periodo_detectado": periodo,
                    # 🆕 #4 — campos estruturados
                    "insights": structured["insights"],
                    "calculations": structured["calculations"],
                    "data_crossings": structured["data_crossings"],
                    "source_references": structured["source_references"],
                    "sources_used": len(docs),
                },
            )

        print(f"[SUMMARY][{job_id}] ✅ DONE | tokens_total={total_tokens_used} | {generation_time_ms}ms")

    except Exception as e:
        print(f"[SUMMARY][{job_id}] ❌ ERROR")
        print(traceback.format_exc())
        job_update(job_id, status="error", error=str(e))


# -------------------------------
# ENRICHMENT
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


# -------------------------------
# UPLOAD (legado — mantido para compatibilidade)
# -------------------------------
@app.post("/upload")
async def upload_documents(project_id: str, files: List[UploadFile] = File(...)):
    try:
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id obrigatório")
        job_id = str(uuid.uuid4())

        async def process_single_file(file: UploadFile):
            content = await file.read()
            # 🆕 #3 — decoder inteligente
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
# ⚡ /process-paths — backend baixa direto do Storage
# -------------------------------
def _download_file(path: str) -> Optional[dict]:
    try:
        data = supabase.storage.from_(BUCKET_NAME).download(path)
        # 🆕 #3 — decoder inteligente (preserva acentos do SPED em latin-1/cp1252)
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
