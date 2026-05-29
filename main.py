from decimal import Decimal
import re
import json
import math
import time
import uuid
import shutil
import asyncio
import logging
import threading
import traceback
import hashlib
from collections import Counter, defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Optional, Dict, Tuple, Any, Set
import os

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, RateLimitError, APIError, AuthenticationError, BadRequestError
from supabase import create_client
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.callbacks.manager import get_openai_callback
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTES BÁSICAS (mantidas do original)
# ==============================================================================
MAX_CHARS_PER_CHUNK = 24000
EMPTY_PLACEHOLDER = " "
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
PERSIST_DIR = "/data/chroma_db"
os.makedirs(PERSIST_DIR, exist_ok=True)
BUCKET_NAME = "sped-documents"
JOBS_TABLE = "backend_jobs"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")
if not OPENAI_API_KEY:    raise Exception("OPENAI_API_KEY não encontrada")
if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise Exception("Credenciais do Supabase não encontradas")
if not INTERNAL_API_KEY:  raise Exception("INTERNAL_API_KEY não configurada")

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH = 256
LLM_MODEL_PLANNER   = "gpt-4o-mini"   # planejamento e crítica → barato
LLM_MODEL_INVESTIGATOR = "gpt-4o-mini"  # loop de retrieval/análise
LLM_MODEL_SYNTH     = "gpt-4o-mini"        # síntese final → qualidade

ALLOWED_LLM_MODELS = {
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
}



def resolve_model(requested: Optional[str], fallback: str) -> str:
    """Valida o modelo pedido pelo cliente; cai no default se inválido."""
    if requested and requested in ALLOWED_LLM_MODELS:
        return requested
    if requested:
        print(f"[MODEL] ⚠️ modelo '{requested}' não permitido, usando '{fallback}'")
    return fallback

openai_client = OpenAI(api_key=OPENAI_API_KEY)
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

# ==============================================================================
# AGENTIC ORCHESTRATION CONFIG
# ==============================================================================
AGENT_MAX_ROUNDS         = 6     # rodadas do loop investigador (cada rodada faz N buscas paralelas)
AGENT_MAX_SEARCHES_TOTAL = 30    # teto absoluto de buscas no Chroma
AGENT_SEARCHES_PER_ROUND = 5     # buscas paralelas por rodada
AGENT_DEFAULT_K          = 8
AGENT_MAX_K              = 20
AGENT_MAX_CHUNK_CHARS    = 1500
AGENT_EVIDENCE_BUDGET    = 250    # máximo de evidências mantidas na memória
AGENT_COMPRESSION_TOPN   = 40    # top-N evidências enviadas ao synthesizer
AGENT_REFLECTION_THRESHOLD = 0.55  # cobertura mínima para encerrar

# ==============================================================================
# UTIL / DECODER (preservado)
# ==============================================================================
def _sanitize_texts(texts: List[str]) -> List[str]:
    cleaned: List[str] = []
    for t in texts:
        if t is None:
            cleaned.append(EMPTY_PLACEHOLDER); continue
        s = str(t).strip() or EMPTY_PLACEHOLDER
        if len(s) > MAX_CHARS_PER_CHUNK:
            s = s[:MAX_CHARS_PER_CHUNK]
        cleaned.append(s)
    return cleaned


def smart_decode_sped(raw: bytes) -> str:
    if not raw: return ""
    if raw.startswith(b"\xef\xbb\xbf"): return raw.decode("utf-8-sig", errors="replace")
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    try:   return raw.decode("utf-8")
    except UnicodeDecodeError: pass
    try:
        decoded = raw.decode("latin-1")
        if any(c in decoded for c in ("\x80","\x82","\x83","\x84","\x85","\x86","\x87","\x88","\x89")):
            try: return raw.decode("cp1252")
            except UnicodeDecodeError: return decoded
        return decoded
    except UnicodeDecodeError: pass
    return raw.decode("utf-8", errors="replace")


# ==============================================================================
# JOB PERSISTENCE
# ==============================================================================
def job_create(job_id, kind, project_id=None, stage="created"):
    supabase.table(JOBS_TABLE).insert({
        "id": job_id, "kind": kind, "project_id": project_id,
        "status": "pending", "stage": stage, "progress": 0,
    }).execute()

def job_update(job_id, **fields):
    allowed = {"status","stage","progress","result","error"}
    payload = {k:v for k,v in fields.items() if k in allowed}
    if not payload: return
    try: supabase.table(JOBS_TABLE).update(payload).eq("id", job_id).execute()
    except Exception as e: print(f"[job_update] erro {job_id}: {e}")

def job_get(job_id):
    try:
        res = supabase.table(JOBS_TABLE).select("*").eq("id", job_id).limit(1).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"[job_get] {e}"); return None

def job_recover_stuck_on_startup():
    try:
        supabase.table(JOBS_TABLE).update({
            "status":"error",
            "error":"Servidor reiniciado durante o processamento. Reenvie.",
        }).in_("status", ["pending","processing"]).execute()
    except Exception as e: print(f"[recover] {e}")


# ==============================================================================
# FASTAPI APP / AUTH
# ==============================================================================
app = FastAPI()

@app.on_event("startup")
def _on_startup(): job_recover_stuck_on_startup()

PUBLIC_ROUTES = ["/status","/docs","/openapi.json"]

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.method == "OPTIONS": return await call_next(request)
    if any(request.url.path.startswith(r) for r in PUBLIC_ROUTES): return await call_next(request)
    if request.headers.get("x-api-key") != INTERNAL_API_KEY:
        return JSONResponse(status_code=401, content={"detail":"Unauthorized"})
    return await call_next(request)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


# ==============================================================================
# EMBEDDINGS (com cache LRU para queries repetidas)
# ==============================================================================
class _LRU(OrderedDict):
    def __init__(self, cap=1024): super().__init__(); self.cap = cap
    def put(self, k, v):
        if k in self: self.move_to_end(k)
        self[k] = v
        if len(self) > self.cap: self.popitem(last=False)

_query_embed_cache = _LRU(2048)


class BatchOpenAIEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        safe = _sanitize_texts(texts); out=[]
        for i in range(0, len(safe), EMBED_BATCH):
            batch = safe[i:i+EMBED_BATCH]; emb=[]
            for attempt in range(5):
                try:
                    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
                    emb = [d.embedding for d in resp.data]
                    if len(emb) != len(batch): raise RuntimeError("mismatch openai embeddings")
                    break
                except AuthenticationError as e: raise RuntimeError("OpenAI API Key inválida.") from e
                except RateLimitError as e:
                    msg=str(e)
                    if "insufficient_quota" in msg: raise RuntimeError("OpenAI sem créditos.") from e
                    if attempt<4: time.sleep(2**attempt); continue
                    raise
                except BadRequestError: raise
                except APIError:
                    if attempt<4: time.sleep(2**attempt); continue
                    raise
                except Exception as e:
                    if ("429" in str(e) or "rate" in str(e).lower()) and attempt<4:
                        time.sleep(2**attempt); continue
                    raise
            out.extend(emb)
        return out

    def embed_query(self, text: str) -> List[float]:
        key = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if key in _query_embed_cache: return _query_embed_cache[key]
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
        v = resp.data[0].embedding
        _query_embed_cache.put(key, v)
        return v


embeddings = BatchOpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


# ==============================================================================
# VECTOR STORE
# ==============================================================================
def get_vector_store(project_id: str):
    try:
        return Chroma(
            collection_name=f"project_{project_id}",
            persist_directory=os.path.join(PERSIST_DIR, project_id),
            embedding_function=embeddings,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro vector store: {e}")


# ==============================================================================
# SPED HELPERS (preservado + tabela de relações entre registros)
# ==============================================================================
SPED_REGISTRO_REGEX = re.compile(r"^\|([0-9A-Z]{4})\|", re.MULTILINE)
SPED_0000_REGEX = re.compile(r"\|0000\|[^|]*\|[^|]*\|(\d{8})\|(\d{8})\|", re.MULTILINE)

# Grafo de relações conhecidas entre registros — usado pelo EvidenceGraph
# para sugerir cruzamentos automáticos.
REGISTRO_RELATIONS: Dict[str, List[str]] = {
    "C100": ["C170","C190","C500","C590"],          # NF-e itens / totais
    "C170": ["C100","C190","0200","M100","M105"],   # itens ↔ produto, créditos PIS
    "C190": ["C100","E110","E111"],                 # totais ICMS
    "C500": ["C590","C595"],                         # NF energia
    "D100": ["D190","D195"],                         # transporte
    "E100": ["E110","E111","E116"],                  # apuração ICMS
    "E110": ["E111","E116","C190","C590","D190"],
    "E200": ["E210","E220","E250"],                  # IPI
    "M100": ["M105","M200","C170"],                  # crédito PIS
    "M200": ["M210","M100","M105"],                  # apuração PIS
    "M500": ["M505","M600","C170"],                  # crédito COFINS
    "M600": ["M610","M500","M505"],                  # apuração COFINS
    "0000": ["0001","0100","0150","0200"],           # cabeçalho
    "0150": ["0200"],
    "0200": ["C170","M100","M105"],                  # cadastro produtos
    "0450": [],
}

# Sinônimos / expansão temática para query expansion
TOPIC_EXPANSIONS: Dict[str, List[str]] = {
    "pis":     ["PIS","M100","M105","M200","M210","CST 50 51 52","crédito presumido","apuração contribuição"],
    "cofins":  ["COFINS","M500","M505","M600","M610","CST 50 51 52","crédito presumido"],
    "icms":    ["ICMS","E100","E110","E111","E116","C190","CFOP","crédito ICMS"],
    "ipi":     ["IPI","E200","E210","E220","E250","C170","CST IPI"],
    "credito": ["crédito tributário","CST 50","insumo","devolução","CFOP 1202 2202 1411"],
    "devolucao":["CFOP 1201 1202 2201 2202","devolução de venda","ajuste base cálculo"],
    "insumo":  ["insumo produção","crédito presumido","C170","NCM","M105","M505"],
    "cfop":    ["CFOP entrada saída","1101 1102 1202","5101 5102 5202","6101 6102"],
    "cst":     ["CST PIS COFINS","CST ICMS","50 51 52 53","60 70 90"],
    "nfe":     ["C100","C170","C190","nota fiscal eletrônica","CFOP NCM"],
    "ncm":     ["NCM mercadoria","classificação fiscal","alíquota"],
    "periodo": ["0000 DT_INI DT_FIN","período apuração"],
    "cnae":    ["CNAE atividade econômica","regime tributário"],
}

def detect_registros(text: str) -> List[str]:
    if not text: return []
    seen, out = set(), []
    for r in SPED_REGISTRO_REGEX.findall(text):
        if r not in seen: seen.add(r); out.append(r)
    return out

def primary_registro(text: str) -> Optional[str]:
    if not text: return None
    m = SPED_REGISTRO_REGEX.findall(text)
    return Counter(m).most_common(1)[0][0] if m else None

def extract_periodo_from_docs(docs) -> Optional[str]:
    ini, fin = [], []
    for d in docs:
        text = getattr(d,"page_content","") or ""
        for m in SPED_0000_REGEX.finditer(text):
            a,b = m.group(1), m.group(2)
            try:
                ini.append(f"{a[0:2]}/{a[2:4]}/{a[4:8]}")
                fin.append(f"{b[0:2]}/{b[2:4]}/{b[4:8]}")
            except: pass
    if not ini or not fin: return None
    si = sorted(ini, key=lambda x:(x[6:10],x[3:5],x[0:2]))
    sf = sorted(fin, key=lambda x:(x[6:10],x[3:5],x[0:2]))
    return f"{si[0]} a {sf[-1]}"

MAX_0450_CHUNKS = 2
def cap_registro_chunks(docs, registro: str, max_keep: int):
    kept, dropped, count = [], 0, 0
    for d in docs:
        reg = d.metadata.get("registro") or primary_registro(d.page_content)
        if reg == registro:
            if count < max_keep: kept.append(d); count+=1
            else: dropped += 1
        else: kept.append(d)
    return kept, dropped

def _filter_sped()->Dict:  return {"source_kind":"sped"}
def _filter_driva()->Dict: return {"source_kind":"driva"}


# ==============================================================================
# DRIVA PRE-PROCESS (preservado)
# ==============================================================================
DRIVA_RELEVANT_KEYS = {
    "cnpj","razao_social","nome_fantasia","natureza_juridica","porte","porte_empresa",
    "capital_social","data_abertura","data_inicio_atividade","situacao_cadastral",
    "situacao","motivo_situacao","regime_tributario","simples_nacional","mei","opcao_simples",
    "opcao_mei","cnae_principal","cnae_fiscal","cnae_fiscal_descricao","cnaes_secundarios",
    "atividade_principal","atividades_secundarias","endereco","municipio","uf","cep",
    "telefone","email","qsa","socios","quadro_socios","matriz_filial","tipo",
}
def _filter_driva_dict(data, depth=0):
    if depth>6: return None
    if isinstance(data, dict):
        out={}
        for k,v in data.items():
            kl=str(k).lower().strip()
            if kl in DRIVA_RELEVANT_KEYS or any(rel in kl for rel in ("cnae","socio","regime","tribut","porte","situacao")):
                f=_filter_driva_dict(v, depth+1)
                if f not in (None,"",[],{}): out[k]=f
        return out
    if isinstance(data, list):
        return [x for x in (_filter_driva_dict(i, depth+1) for i in data[:10]) if x not in (None,"",[],{})]
    return data

def preprocess_driva(en: Dict)->Dict:
    if not isinstance(en, dict): return {}
    f = _filter_driva_dict(en)
    return f if isinstance(f, dict) else {}


# ==============================================================================
# HYBRID RETRIEVER  (semantic + BM25 over candidate pool + metadata + rerank)
# ==============================================================================
_WORD_RE = re.compile(r"[A-Za-zÀ-ú0-9]{2,}")

def _tokenize(s: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(s or "")]

def _bm25_score(query_tokens: List[str], doc_tokens: List[str], avgdl: float, doc_freqs: Dict[str,int], N: int, k1=1.5, b=0.75) -> float:
    if not doc_tokens: return 0.0
    score=0.0; dl=len(doc_tokens)
    tf = Counter(doc_tokens)
    for q in set(query_tokens):
        if q not in doc_freqs: continue
        df = doc_freqs[q]
        idf = math.log(1 + (N - df + 0.5)/(df + 0.5))
        f = tf.get(q,0)
        score += idf * (f*(k1+1)) / (f + k1*(1 - b + b*dl/max(avgdl,1)))
    return score

def _expand_query(query: str) -> str:
    """Query expansion temática: anexa sinônimos/registros conhecidos quando o
    tópico aparece na query, melhorando o recall semântico."""
    ql = query.lower()
    extra: List[str] = []
    for topic, exps in TOPIC_EXPANSIONS.items():
        if topic in ql:
            extra.extend(exps)
    # registros explícitos (ex: M100) também viram âncoras
    for reg in re.findall(r"\b[A-Z]\d{3}\b", query):
        extra.append(reg)
    if not extra: return query
    return f"{query} | {' '.join(sorted(set(extra)))}"

def hybrid_search(
    project_id: str,
    query: str,
    k: int = AGENT_DEFAULT_K,
    registro: Optional[str] = None,
    source_kind: str = "sped",
    fetch_multiplier: int = 4,
) -> List[Any]:
    """
    Pipeline de retrieval híbrido:
      1. Query expansion temática.
      2. Semântica (MMR) sobre vector store com pool grande.
      3. BM25 reranking dentro do pool.
      4. Score combinado (0.6 semântico-rank + 0.4 BM25-norm).
      5. Filtro por registro (se solicitado).
      6. Cap especial registro 0450.
    """
    vs = get_vector_store(project_id)
    expanded = _expand_query(query)
    pool_k = min(max(k * 2, 10), 25)

    flt: Dict[str, Any] = {"source_kind": source_kind}
    if registro: flt["registro"] = registro

    try:
        pool = vs.max_marginal_relevance_search(expanded, k=pool_k, fetch_k=pool_k*2, filter=flt)
    except Exception:
        try:    pool = vs.max_marginal_relevance_search(expanded, k=pool_k, fetch_k=pool_k*2)
        except Exception as e:
            print(f"[hybrid_search] vector falhou: {e}"); return []

    if not pool: return []

    # BM25 rerank
    docs_tokens = [_tokenize(d.page_content) for d in pool]
    q_tokens = _tokenize(expanded)
    N = len(docs_tokens)
    avgdl = sum(len(t) for t in docs_tokens) / max(N, 1)
    doc_freqs: Dict[str,int] = defaultdict(int)
    for toks in docs_tokens:
        for t in set(toks): doc_freqs[t] += 1
    bm25_scores = [_bm25_score(q_tokens, t, avgdl, doc_freqs, N) for t in docs_tokens]
    bm25_max = max(bm25_scores) or 1.0
    bm25_norm = [s/bm25_max for s in bm25_scores]

    # rank semântico = posição no pool (MMR já ordenou por relevância+diversidade)
    sem_norm = [1.0 - (i/max(N-1,1)) for i in range(N)]

    combined = [(0.6*sem_norm[i] + 0.4*bm25_norm[i], pool[i]) for i in range(N)]
    combined.sort(key=lambda x: x[0], reverse=True)
    ranked = [d for _, d in combined[:k]]

    ranked, _ = cap_registro_chunks(ranked, "0450", MAX_0450_CHUNKS)
    return ranked


# ==============================================================================
# INVESTIGATION MEMORY + EVIDENCE GRAPH
# ==============================================================================
class InvestigationMemory:
    """
    Memória persistente da investigação (vive durante a geração de UM sumário).
    Guarda: queries feitas, evidências coletadas (deduplicadas), hipóteses,
    lacunas detectadas, registros já cobertos.
    """
    def __init__(self):
        self.queries_done: List[str] = []
        self.queries_set: Set[str] = set()
        self.evidence: List[Dict[str, Any]] = []   # cada item: source/registro/text/score/topic
        self.evidence_keys: Set[str] = set()
        self.registros_covered: Counter = Counter()
        self.hypotheses: List[Dict[str, Any]] = [] # {topic, status:open|confirmed|rejected, note}
        self.gaps: List[str] = []                  # descrições textuais de lacunas
        self.topics_covered: Set[str] = set()

    def has_query(self, q: str) -> bool:
        return q.strip().lower() in self.queries_set

    def add_query(self, q: str):
        ql = q.strip().lower()
        self.queries_set.add(ql); self.queries_done.append(q)

    def add_evidence(self, doc: Any, topic: str, score: float):
        text = (doc.page_content or "")[:AGENT_MAX_CHUNK_CHARS]
        key = hashlib.sha1(text.encode("utf-8","ignore")).hexdigest()
        if key in self.evidence_keys: return False
        self.evidence_keys.add(key)
        reg = doc.metadata.get("registro") or primary_registro(doc.page_content) or "?"
        self.evidence.append({
            "source":   doc.metadata.get("source","?"),
            "chunk":    doc.metadata.get("chunk_index"),
            "registro": reg,
            "kind":     doc.metadata.get("source_kind","?"),
            "topic":    topic,
            "score":    round(float(score), 4),
            "text":     text,
        })
        self.registros_covered[reg] += 1
        # evict pior se exceder budget
        if len(self.evidence) > AGENT_EVIDENCE_BUDGET:
            self.evidence.sort(key=lambda e: e["score"], reverse=True)
            self.evidence = self.evidence[:AGENT_EVIDENCE_BUDGET]
            self.evidence_keys = {hashlib.sha1(e["text"].encode("utf-8","ignore")).hexdigest() for e in self.evidence}
        return True

    def add_hypothesis(self, topic: str, note: str = ""):
        if not any(h["topic"]==topic for h in self.hypotheses):
            self.hypotheses.append({"topic": topic, "status":"open", "note":note})

    def update_hypothesis(self, topic: str, status: str, note: str = ""):
        for h in self.hypotheses:
            if h["topic"]==topic:
                h["status"]=status
                if note: h["note"]=note
                return

    def mark_topic(self, topic: str): self.topics_covered.add(topic.lower())

    def coverage_score(self, planned_topics: List[str]) -> float:
        if not planned_topics: return 1.0
        hit = sum(1 for t in planned_topics if t.lower() in self.topics_covered)
        return hit / len(planned_topics)

    def top_evidence(self, n: int) -> List[Dict[str, Any]]:
        return sorted(self.evidence, key=lambda e: e["score"], reverse=True)[:n]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "queries":         self.queries_done,
            "evidence_count":  len(self.evidence),
            "registros":       dict(self.registros_covered),
            "topics_covered":  sorted(self.topics_covered),
            "hypotheses":      self.hypotheses,
            "gaps":            self.gaps,
        }


class EvidenceGraph:
    """
    Liga evidências por registro via REGISTRO_RELATIONS.
    Produz triplos investigativos: (registro_a) -[relates]-> (registro_b)
    e ajuda o synthesizer a explicar cruzamentos.
    """
    def __init__(self): self.edges: Set[Tuple[str,str]] = set()

    def ingest(self, mem: InvestigationMemory):
        regs = set(mem.registros_covered.keys())
        for r in regs:
            for rel in REGISTRO_RELATIONS.get(r, []):
                if rel in regs:
                    a,b = sorted([r,rel])
                    self.edges.add((a,b))

    def suggest_next_registros(self, mem: InvestigationMemory, top: int = 3) -> List[str]:
        """Sugere registros faltantes que se conectam aos já cobertos."""
        cov = set(mem.registros_covered.keys())
        cand: Counter = Counter()
        for r in cov:
            for rel in REGISTRO_RELATIONS.get(r, []):
                if rel not in cov: cand[rel] += 1
        return [r for r,_ in cand.most_common(top)]

    def export(self) -> List[Dict[str,str]]:
        return [{"from":a, "to":b} for (a,b) in sorted(self.edges)]


# ==============================================================================
# CONTEXT COMPRESSOR
# ==============================================================================
class SpedAnalyticsEngine:

    def __init__(self, evidence: List[Dict[str, Any]]):

        self.evidence = evidence

        self.metrics = {
            "c100_vs_c170": {
                "documents_analyzed": 0,
                "divergent_documents": 0,
                "total_divergence": 0.0,
            },

            "ncm_inconsistencies": {
                "count": 0,
                "affected_ncms": [],
            },

            "cfop_statistics": {},

            "cst_statistics": {},

            "credit_analysis": {
                "pis_credit_total": 0.0,
                "cofins_credit_total": 0.0,
                "credit_documents": 0,
            },

            "sequence_breaks": {
                "count": 0,
            },

            "registros": {},

            "totals": {
                "evidence_count": len(evidence),
            }
        }

    # =========================================================
    # HELPERS
    # =========================================================

    def _extract_money_values(self, text: str) -> List[float]:

        vals = []

        for m in re.findall(r"\d{1,3}(?:\.\d{3})*,\d{2}", text):

            try:
                vals.append(
                    float(
                        m.replace(".", "").replace(",", ".")
                    )
                )
            except:
                pass

        return vals

    def _extract_cfops(self, text: str) -> List[str]:

        return re.findall(r"\b([1-7]\d{3})\b", text)

    def _extract_csts(self, text: str) -> List[str]:

        return re.findall(r"\bCST[\s:]*([0-9]{2})\b", text, flags=re.I)

    def _extract_ncms(self, text: str) -> List[str]:

        return re.findall(r"\b(\d{8})\b", text)

    # =========================================================
    # REGISTRO COUNTS
    # =========================================================

    def build_registro_metrics(self):

        counter = Counter()

        for ev in self.evidence:

            reg = ev.get("registro")

            if reg:
                counter[reg] += 1

        self.metrics["registros"] = dict(counter)

    # =========================================================
    # CFOP ANALYTICS
    # =========================================================

    def build_cfop_metrics(self):

        cfop_counter = Counter()

        for ev in self.evidence:

            text = ev.get("text", "")

            cfops = self._extract_cfops(text)

            for cfop in cfops:
                cfop_counter[cfop] += 1

        self.metrics["cfop_statistics"] = dict(cfop_counter)

    # =========================================================
    # CST ANALYTICS
    # =========================================================

    def build_cst_metrics(self):

        cst_counter = Counter()

        for ev in self.evidence:

            text = ev.get("text", "")

            csts = self._extract_csts(text)

            for cst in csts:
                cst_counter[cst] += 1

        self.metrics["cst_statistics"] = dict(cst_counter)

    # =========================================================
    # NCM ANALYTICS
    # =========================================================

    def build_ncm_metrics(self):

        ncms = []

        for ev in self.evidence:

            text = ev.get("text", "")

            found = self._extract_ncms(text)

            ncms.extend(found)

        counter = Counter(ncms)

        inconsistents = []

        for ncm, count in counter.items():

            if count >= 3:
                inconsistents.append(ncm)

        self.metrics["ncm_inconsistencies"] = {
            "count": len(inconsistents),
            "affected_ncms": inconsistents[:30],
        }

    # =========================================================
    # CREDIT ANALYSIS
    # =========================================================

    def build_credit_metrics(self):

        pis_total = 0.0
        cofins_total = 0.0
        docs = 0

        for ev in self.evidence:

            text = ev.get("text", "").lower()

            values = self._extract_money_values(text)

            if not values:
                continue

            if "pis" in text:

                pis_total += sum(values)
                docs += 1

            if "cofins" in text:

                cofins_total += sum(values)
                docs += 1

        self.metrics["credit_analysis"] = {
            "pis_credit_total": round(pis_total, 2),
            "cofins_credit_total": round(cofins_total, 2),
            "credit_documents": docs,
        }

    # =========================================================
    # C100 x C170
    # =========================================================

    def build_c100_c170_metrics(self):

        c100_docs = []
        c170_docs = []

        for ev in self.evidence:

            reg = ev.get("registro")

            if reg == "C100":
                c100_docs.append(ev)

            elif reg == "C170":
                c170_docs.append(ev)

        divergences = 0
        divergence_total = 0.0

        c170_values = []

        for ev in c170_docs:

            vals = self._extract_money_values(
                ev.get("text", "")
            )

            c170_values.extend(vals)

        c100_values = []

        for ev in c100_docs:

            vals = self._extract_money_values(
                ev.get("text", "")
            )

            c100_values.extend(vals)

        compare_count = min(
            len(c100_values),
            len(c170_values)
        )

        for i in range(compare_count):

            diff = abs(
                c100_values[i] - c170_values[i]
            )

            if diff > 0.5:

                divergences += 1
                divergence_total += diff

        self.metrics["c100_vs_c170"] = {
            "documents_analyzed": compare_count,
            "divergent_documents": divergences,
            "total_divergence": round(divergence_total, 2),
        }

    # =========================================================
    # BUILD ALL
    # =========================================================

    def build_metrics(self):

        self.build_registro_metrics()

        self.build_cfop_metrics()

        self.build_cst_metrics()

        self.build_ncm_metrics()

        self.build_credit_metrics()

        self.build_c100_c170_metrics()

        return self.metrics

def build_analytics(mem: InvestigationMemory) -> Dict[str, Any]:

    engine = SpedAnalyticsEngine(
        mem.evidence
    )

    analytics = engine.build_metrics()

    # =========================================================
    # TOPICS
    # =========================================================

    topic_counter = Counter()

    for ev in mem.evidence:

        topic = ev.get("topic")

        if topic:
            topic_counter[topic] += 1

    analytics["topics"] = {}

    for topic, count in topic_counter.items():

        analytics["topics"][topic] = {
            "evidence_count": count
        }

    # =========================================================
    # SOURCES
    # =========================================================

    source_counter = Counter()

    for ev in mem.evidence:

        src = ev.get("source")

        if src:
            source_counter[src] += 1

    analytics["sources"] = dict(source_counter)

    return analytics

# ==============================================================================
# MONOLITHIC DETERMINISTIC ANALYSIS ENGINE
# ==============================================================================
# Tudo fica no main.py: calculos, financeiro, juridico, BI, scoring e schema final.
# O LLM recebe apenas o JSON calculado e gera narrativa.

MATERIALITY_THRESHOLDS = {"HIGH": Decimal("100000.00"), "MEDIUM": Decimal("25000.00")}
LEGAL_RULE_DATABASE = [
    {"terms": ("PIS", "COFINS", "M100", "M500"), "law": "Lei 10.833/2003", "article": "Art. 3", "context": "Creditamento PIS/COFINS"},
    {"terms": ("ICMS", "E110", "E111", "C190"), "law": "LC 87/1996", "article": "Arts. 19 a 23", "context": "Apuracao e creditos de ICMS"},
    {"terms": ("C100", "C170", "C190"), "law": "Ajuste SINIEF 02/2009", "article": "Leiaute EFD", "context": "Consistencia documento/item/totalizador"},
    {"terms": ("NCM", "0200", "C170"), "law": "TIPI/NCM vigente", "article": "RGI-SH", "context": "Classificacao fiscal"},
]

def _to_decimal(value: Any, default: str = "0") -> Decimal:
    try:
        if isinstance(value, Decimal): return value
        s = str(value or "").strip().replace("R$", "").replace(" ", "")
        if not s: return Decimal(default)
        if "," in s: s = s.replace(".", "").replace(",", ".")
        return Decimal(s)
    except Exception:
        return Decimal(default)

def _money_float(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01")))

def _risk_level(score: float) -> str:
    return "HIGH" if score >= 75 else "MEDIUM" if score >= 45 else "LOW"

def _materiality_level(amount: Decimal) -> str:
    return "HIGH" if amount >= MATERIALITY_THRESHOLDS["HIGH"] else "MEDIUM" if amount >= MATERIALITY_THRESHOLDS["MEDIUM"] else "LOW"

def _extract_decimal_values(text: str) -> List[Decimal]:
    return [_to_decimal(x) for x in re.findall(r"(?:R\$\s*)?\d{1,3}(?:\.\d{3})*,\d{2}|\b\d+\.\d{2}\b", text or "")]

def normalize_sped_data(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    records, by_register = [], defaultdict(list)
    for ev in evidence or []:
        raw = ev.get("text", "") or ""
        reg = ev.get("registro") or primary_registro(raw) or "unknown"
        parsed = {"money_values": _extract_decimal_values(raw), "cfops": re.findall(r"\b([1-7]\d{3})\b", raw), "csts": re.findall(r"\bCST[\s:]*([0-9]{2})\b", raw, flags=re.I), "ncms": re.findall(r"\b(\d{8})\b", raw)}
        rec = {"source": ev.get("source", "unknown"), "registro": reg, "topic": ev.get("topic", "geral"), "chunk": ev.get("chunk"), "score": float(ev.get("score") or 0), "text": raw[:AGENT_MAX_CHUNK_CHARS], "parsed": parsed}
        records.append(rec); by_register[reg].append(rec)
    return {"records": records, "by_register": dict(by_register), "record_count": len(records), "register_count": len(by_register)}

def run_deterministic_analysis(normalized: Dict[str, Any]) -> Dict[str, Any]:
    amounts, cfops, csts, ncms = defaultdict(lambda: Decimal("0")), Counter(), Counter(), Counter()
    total, findings = Decimal("0"), []
    for rec in normalized.get("records", []):
        vals = rec["parsed"].get("money_values", [])
        subtotal = sum(vals, Decimal("0")); total += subtotal; amounts[rec["registro"]] += subtotal
        cfops.update(rec["parsed"].get("cfops", [])); csts.update(rec["parsed"].get("csts", [])); ncms.update(rec["parsed"].get("ncms", []))
    c100, c170 = amounts.get("C100", Decimal("0")), amounts.get("C170", Decimal("0"))
    diff = abs(c100 - c170)
    if c100 and c170 and diff > Decimal("0.50"):
        findings.append({"finding": "Divergencia agregada C100/C170", "source_registers": ["C100", "C170"], "documents_affected": len(normalized.get("by_register", {}).get("C100", [])), "calculation_method": "abs(sum(C100) - sum(C170))", "amount": _money_float(diff), "confidence_score": 0.72})
    recurrent_ncms = [n for n, c in ncms.items() if c >= 3]
    if recurrent_ncms:
        findings.append({"finding": "NCMs recorrentes para revisao", "source_registers": ["0200", "C170"], "documents_affected": len(recurrent_ncms), "calculation_method": "count(NCM) >= 3", "affected_ncms": recurrent_ncms[:20], "confidence_score": 0.68})
    count = max(normalized.get("record_count", 0), 1)
    confidence = round(min(1, count / 40) * 0.40 + min(1, len(cfops) / 10) * 0.25 + (1 - min(1, len(findings) / count)) * 0.25 + (0.10 if total else 0.02), 2)
    return {"totals": {"records_analyzed": normalized.get("record_count", 0), "registers_analyzed": normalized.get("register_count", 0), "monetary_volume_detected": _money_float(total), "amount_by_register": {k: _money_float(v) for k, v in sorted(amounts.items())}}, "cross_validations": {"c100_total": _money_float(c100), "c170_total": _money_float(c170), "c100_c170_difference": _money_float(diff)}, "tax_indicators": {"top_cfops": cfops.most_common(20), "top_csts": csts.most_common(20), "top_ncms": ncms.most_common(20)}, "findings": findings, "confidence_score": confidence}

def calculate_financial_summary(deterministic: Dict[str, Any]) -> Dict[str, Any]:
    volume = _to_decimal(deterministic.get("totals", {}).get("monetary_volume_detected")); diff = _to_decimal(deterministic.get("cross_validations", {}).get("c100_c170_difference"))
    credit_base = volume * Decimal("0.0925") if volume else Decimal("0"); exposure = max(diff, credit_base * Decimal("0.15"))
    return {"estimated_tax_exposure": _money_float(exposure), "annual_projection": _money_float(exposure * Decimal("12")), "materiality_level": _materiality_level(exposure), "estimated_recoverable_amount": _money_float(credit_base * Decimal("0.30")), "confidence_level": round(float(deterministic.get("confidence_score") or 0.1), 2), "calculation_methods": {"estimated_tax_exposure": "max(C100/C170 difference, monetary_volume * 9.25% * 15%)", "annual_projection": "estimated_tax_exposure * 12", "estimated_recoverable_amount": "monetary_volume * 9.25% * 30%"}}

def run_legal_analysis(normalized: Dict[str, Any], deterministic: Dict[str, Any]) -> Dict[str, Any]:
    haystack = " ".join((r.get("registro", "") + " " + r.get("text", "")[:300]) for r in normalized.get("records", [])).upper()
    laws = [{"law": r["law"], "article": r["article"], "context": r["context"], "matched_terms": [t for t in r["terms"] if t.upper() in haystack]} for r in LEGAL_RULE_DATABASE if any(t.upper() in haystack for t in r["terms"])]
    if not laws: laws = [{"law": "SPED/EFD", "article": "Leiaute aplicavel", "context": "Validacao dos registros extraidos", "matched_terms": sorted(normalized.get("by_register", {}).keys())[:8]}]
    risk = min(100, 35 + len(deterministic.get("findings", [])) * 12 + len(laws) * 4)
    return {"applicable_laws": laws, "litigation_risk": _risk_level(risk), "defensibility_score": max(0, 100 - risk + int((deterministic.get("confidence_score") or 0) * 25)), "legal_confidence": round(float(deterministic.get("confidence_score") or 0.1), 2)}

def build_business_intelligence(deterministic: Dict[str, Any], analytics: Dict[str, Any]) -> Dict[str, Any]:
    ind = deterministic.get("tax_indicators", {}); amounts = deterministic.get("totals", {}).get("amount_by_register", {})
    return {"top_ncms": ind.get("top_ncms", [])[:10], "top_cfops": ind.get("top_cfops", [])[:10], "top_csts": ind.get("top_csts", [])[:10], "top_registers_by_value": sorted(amounts.items(), key=lambda x: x[1], reverse=True)[:10], "sources": analytics.get("sources", {}), "topics": analytics.get("topics", {})}

def calculate_scores(financial: Dict[str, Any], legal: Dict[str, Any], deterministic: Dict[str, Any]) -> Dict[str, Any]:
    materiality = 95 if financial.get("materiality_level") == "HIGH" else 65 if financial.get("materiality_level") == "MEDIUM" else 30
    divergence = min(100, float(_to_decimal(deterministic.get("cross_validations", {}).get("c100_c170_difference"))) / 1000)
    recurrence = min(100, len(deterministic.get("tax_indicators", {}).get("top_cfops", [])) * 5)
    legal_score = 80 if legal.get("litigation_risk") == "HIGH" else 55 if legal.get("litigation_risk") == "MEDIUM" else 25
    overall = divergence * 0.25 + materiality * 0.25 + recurrence * 0.15 + legal_score * 0.20 + float(financial.get("confidence_level") or 0) * 100 * 0.15
    return {"overall_risk_score": int(round(overall)), "financial_risk_score": materiality, "compliance_score": max(0, 100 - int(overall * 0.55)), "partner_risk_score": 0, "risk_level": _risk_level(overall), "weights": {"tax_divergence": 0.25, "financial_materiality": 0.25, "recurrence_factor": 0.15, "legal_risk": 0.20, "confidence": 0.15}}

def build_final_analysis_schema(mem: InvestigationMemory, analytics: Dict[str, Any], graph: EvidenceGraph, periodo: Optional[str]) -> Dict[str, Any]:
    normalized = normalize_sped_data(mem.evidence); deterministic = run_deterministic_analysis(normalized)
    financial = calculate_financial_summary(deterministic); legal = run_legal_analysis(normalized, deterministic); bi = build_business_intelligence(deterministic, analytics); scoring = calculate_scores(financial, legal, deterministic)
    evidence = [{"source": e.get("source"), "registro": e.get("registro"), "topic": e.get("topic"), "chunk": e.get("chunk"), "confidence_score": round(float(e.get("score") or 0), 2)} for e in mem.evidence[:AGENT_EVIDENCE_BUDGET]]
    return {"metadata": {"periodo": periodo or "periodo_nao_detectado", "engine_mode": "monolithic_main_py", "llm_role": "narrative_only", "records_analyzed": normalized.get("record_count", 0)}, "financial_layer": financial, "legal_layer": legal, "business_intelligence": bi, "scoring": scoring, "deterministic_analysis": deterministic, "narrative": {"status": "pending_llm_generation"}, "evidence": evidence, "graph": graph.export()}


def compress_context(evidence: List[Dict[str,Any]], topn: int = AGENT_COMPRESSION_TOPN) -> str:
    """
    Compressão de contexto:
      - Top-N por score
      - Dedup textual (já garantido na memória)
      - Trunca cada chunk
      - Agrupa por (source, registro) para reduzir ruído
    """
    if not evidence: return "(sem evidências)"
    top = sorted(evidence, key=lambda e: e["score"], reverse=True)[:topn]
    by_src: Dict[Tuple[str,str], List[Dict[str,Any]]] = defaultdict(list)
    for e in top:
        by_src[(e["source"], e["registro"])].append(e)
    parts: List[str] = []
    for (src, reg), items in by_src.items():
        head = f"### Fonte: {src} | Registro: {reg} | {len(items)} evidência(s)"
        body = "\n\n".join(
            f"[score={it['score']} | topic={it['topic']} | chunk={it['chunk']}]\n{it['text']}"
            for it in items
        )
        parts.append(f"{head}\n{body}")
    return "\n\n---\n\n".join(parts)


# ==============================================================================
# RETRIEVAL ROUTER
# ==============================================================================
def route_query(query: str) -> str:
    """Decide se a query deve ir para SPED ou Driva."""
    ql = query.lower()
    driva_terms = ("cnae","porte","sócio","socio","quadro societário","regime tributário",
                   "situação cadastral","capital social","atividade econômica","razão social")
    if any(t in ql for t in driva_terms): return "driva"
    return "sped"


# ==============================================================================
# LLM AGENTS — PLANNER / CRITIC / SYNTHESIZER
# ==============================================================================
def _llm_json(model: str, system: str, user: str, max_tokens: int = 1200) -> Dict[str, Any]:
    """Chamada LLM forçando JSON."""
    resp = openai_client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":system},{"role":"user","content":user}],
        temperature=0.1,
        response_format={"type":"json_object"},
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content or "{}"
    try: data = json.loads(raw)
    except Exception: data = {"_raw": raw}
    data["_usage"] = {
        "prompt_tokens":     resp.usage.prompt_tokens if resp.usage else 0,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
    }
    return data


PLANNER_SYS = """Você é o PLANNER de um sistema agentic-RAG fiscal/SPED.
Sua tarefa: a partir do TEMPLATE do usuário, produzir um PLANO de investigação.

Retorne JSON estrito:
{
  "topics":       [ "string curta (ex: 'créditos PIS sobre devoluções')", ... ],   // 5-12 tópicos
  "registros":    [ "M100","C170", ... ],                                          // registros SPED prováveis
  "hypotheses":   [ {"topic":"...", "note":"...por que vale investigar"} ],        // 3-8 hipóteses
  "initial_queries": [ "query semântica concreta", ... ],                          // 4-8 queries iniciais
  "needs_driva":  true|false                                                       // se contexto da empresa importa
}

Regras:
- topics devem mapear seções/itens do template.
- queries devem ser concretas, com termos fiscais e códigos de registro.
- não invente dados; apenas planeje O QUE buscar.
"""

CRITIC_SYS = """Você é o CRITIC de um sistema agentic-RAG fiscal.
Recebe: o plano original, a memória de investigação (queries+evidências+registros cobertos)
e a cobertura atual dos tópicos.

Tarefa: decidir se a investigação está pronta para sintetizar OU se precisa de mais buscas.

Retorne JSON estrito:
{
  "ready":       true|false,                       // pronto para sintetizar?
  "gaps":        [ "descrição da lacuna", ... ],   // lacunas a cobrir
  "refined_queries": [ "query semântica refinada", ... ],  // até 5; novas, não repetir
  "target_registros": [ "M200", ... ],             // registros que ainda devem ser buscados
  "reasoning":   "1-2 frases curtas"
}

Regras:
- Se cobertura de tópicos >= 0.7 E há evidência por tópico, marque ready=true.
- Não repita queries já feitas (lista fornecida).
- Prefira refinar (sinônimos, código de registro, CFOP/CST) em vez de repetir.
"""

SYNTH_SYS = """
Você é o SYNTHESIZER fiscal/SPED responsável por produzir relatórios fiscais profissionais em Markdown renderizável via ReactMarkdown.


Priorize SEMPRE:
1. analytics estruturados
2. métricas quantitativas
3. agregações
4. cruzamentos calculados

Responda SOMENTE com o documento final.
NÃO estime valores.
NÃO estime impactos financeiros.
NÃO estime quantidade de ocorrências.
NÃO use faixas aproximadas:
"3 a 5"
"10+"
"aproximadamente"
"cerca de"
"estimado"
"potencial"
NÃO classifique risco como:
baixo
médio
alto
sem evidência explícita.
NÃO deduza métricas a partir do contexto.
NÃO extrapole dados fiscais.
Só escreva números que estejam explicitamente presentes nas evidências recebidas.
Se um valor não existir literalmente nas evidências:
OMITA COMPLETAMENTE.
NÃO explique seu raciocínio.
NÃO explique sua função.
NÃO repita instruções recebidas.
NÃO repita prompts do sistema.
NÃO escreva introduções como:
"Claro", "Segue abaixo", "Aqui está o relatório", etc.

NÃO envolva a resposta em blocos:
ou qualquer cerca de código.

Retorne Markdown PURO.
Use apenas:
headings markdown (## ### ####)
listas markdown
tabelas markdown válidas
negrito/itálico markdown
O conteúdo deve ser renderizável diretamente pelo ReactMarkdown + remark-gfm.
NÃO use HTML.
NÃO use .
NÃO use tags XML.
NÃO use JSON.
NÃO use YAML.
NÃO use backticks triplos em nenhuma circunstância.

Você receberá:

Template original do usuário
Período fiscal
Evidências comprimidas (top-N)
Grafo de cruzamentos detectados
Hipóteses investigadas
Lacunas conhecidas

Produzir o SUMÁRIO FINAL obedecendo ESTRITAMENTE o template solicitado.

REGRAS CRÍTICAS DE EVIDÊNCIA:

- NÃO escreva:
  "Não estimado"
  "Não avaliado"
  "Indefinido"
  "Valores variados"
  "Diversas ocorrências"
  "Não disponível"

- Se não houver dado objetivo:
  OMITA COMPLETAMENTE a linha, coluna ou item.

- NÃO invente ranges aproximados.
- NÃO extrapole impacto financeiro.
- NÃO estime ROI sem cálculo explícito nas evidências.
- NÃO gere tabelas parcialmente vazias.

- Só inclua tabelas se houver pelo menos:
  - 2 linhas completas
  - números reais
  - evidência concreta

- Se uma seção não possuir evidência suficiente:
  remova a seção inteira do output final.   

NÃO invente números.
Toda afirmação numérica/fiscal deve estar baseada em evidências reais.
Sempre que possível, cite:
arquivo
registro SPED
CFOP/CST/NCM relevantes
Se NÃO houver evidência suficiente:
OMITA o item
NÃO escreva:
"Dado não disponível"
"Informação insuficiente"
placeholders vazios
Dados da Driva:
servem apenas como contexto empresarial
nunca como prova numérica/fiscal
Registro 0450:
é apenas informação complementar
não utilizar para cálculos de ROI ou créditos
Sempre utilize o período fiscal informado.
Sempre que cruzar múltiplos registros:
explique claramente a relação entre eles
Substitua placeholders como:
X.N
Capítulo X
Seção X
por numeração real.
Linguagem técnica e profissional.
Objetiva.
Sem redundância.
Sem repetir evidências iguais.
Sem repetir tabelas similares.
Priorize clareza executiva.
Use tabelas apenas quando agregarem valor real.

A resposta final deve começar DIRETAMENTE no conteúdo do relatório.

Exemplo correto:

1. Sumário Executivo

Texto...

2. Créditos de PIS/COFINS

Tabela...

Exemplo incorreto:

## 1. Sumário Executivo

ou

"Segue abaixo o relatório..."
"""


def planner_agent(template: str, query: str, model: str) -> Dict[str, Any]:
    user = f"TEMPLATE:\n\"\"\"\n{template[:6000]}\n\"\"\"\n\nOBJETIVO: {query}\n"
    out = _llm_json(model, PLANNER_SYS, user, max_tokens=1200)
    out.setdefault("topics", []); out.setdefault("registros", [])
    out.setdefault("hypotheses", []); out.setdefault("initial_queries", [])
    out.setdefault("needs_driva", True)
    return out


def critic_agent(plan: Dict[str, Any], mem: InvestigationMemory,
                 graph: EvidenceGraph, model: str) -> Dict[str, Any]:
    coverage = mem.coverage_score(plan.get("topics", []))
    payload = {
        "plan_topics":       plan.get("topics", []),
        "plan_registros":    plan.get("registros", []),
        "queries_done":      mem.queries_done[-20:],
        "registros_covered": dict(mem.registros_covered),
        "topics_covered":    sorted(mem.topics_covered),
        "coverage_score":    round(coverage, 3),
        "evidence_count":    len(mem.evidence),
        "graph_edges":       graph.export(),
        "hypotheses":        mem.hypotheses,
    }
    out = _llm_json(
        model, CRITIC_SYS,
        json.dumps(payload, ensure_ascii=False), max_tokens=700,
    )
    out.setdefault("ready", coverage >= AGENT_REFLECTION_THRESHOLD and len(mem.evidence) > 0)
    out.setdefault("gaps", []); out.setdefault("refined_queries", [])
    out.setdefault("target_registros", []); out.setdefault("reasoning", "")
    return out


def synthesizer_agent(template: str, periodo: Optional[str],
                      mem: InvestigationMemory,
                      analytics: Dict[str, Any], graph: EvidenceGraph,
                      model: str) -> Tuple[str, Dict[str, int]]:
    compressed = compress_context(mem.evidence, AGENT_COMPRESSION_TOPN)
    user = f"""PERÍODO FISCAL: {periodo or 'não detectado'}

TEMPLATE:
\"\"\"
{template}
\"\"\"

CRUZAMENTOS DETECTADOS (evidence graph):
{json.dumps(graph.export(), ensure_ascii=False)}

HIPÓTESES INVESTIGADAS:
{json.dumps(mem.hypotheses, ensure_ascii=False)}

LACUNAS CONHECIDAS:
{json.dumps(mem.gaps, ensure_ascii=False)}

MÉTRICAS FISCAIS ESTRUTURADAS:
{json.dumps(analytics, ensure_ascii=False, indent=2)}

EVIDÊNCIAS QUALITATIVAS:
{compressed}

Produza AGORA o sumário final em Markdown.
"""
    resp = openai_client.chat.completions.create(
        model=model,                          # ⬅️ dinâmico
        messages=[{"role": "system", "content": SYNTH_SYS},
                  {"role": "user", "content": user}],
        temperature=0.2,
    )
    usage = {
        "prompt_tokens":     resp.usage.prompt_tokens if resp.usage else 0,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
    }
    return resp.choices[0].message.content or "", usage


# ==============================================================================
# ORCHESTRATOR  —  Planner → Investigator loop (parallel + reflection) → Synth
# ==============================================================================
def _run_search_task(args: Tuple[str,str,int,Optional[str],str]) -> Tuple[str, List[Any], str]:
    project_id, query, k, registro, topic = args
    kind = route_query(query)
    docs = hybrid_search(project_id, query, k=k, registro=registro, source_kind=kind)
    return query, docs, topic


def orchestrate_summary(req: "SummaryRequest", job_id: str) -> Dict[str, Any]:
    project_id = req.project_id
    template = req.template
    objective = req.query or "gerar sumário fiscal completo"
    
    model_planner = resolve_model(req.model, LLM_MODEL_PLANNER)
    model_critic  = resolve_model(req.model, LLM_MODEL_INVESTIGATOR)
    model_synth   = resolve_model(req.model, LLM_MODEL_SYNTH)
    print(f"[ORCH][{job_id}] models => planner={model_planner} critic={model_critic} synth={model_synth}")

    t0 = time.time()
    tokens_total = {"prompt":0, "completion":0}
    trace: List[Dict[str, Any]] = []

    # ---------- FASE 1: PLAN ----------
    job_update(job_id, stage="agent_plan")
    plan = planner_agent(
    template=template,
    query=objective,
    model=model_planner,
    )
    u = plan.pop("_usage", {})
    tokens_total["prompt"]     += u.get("prompt_tokens", 0)
    tokens_total["completion"] += u.get("completion_tokens", 0)
    
    trace.append({"phase": "plan", "topics": plan["topics"],
                  "registros": plan["registros"], "queries": plan["initial_queries"]})
    print(f"[ORCH][{job_id}] PLAN | topics={len(plan['topics'])} regs={plan['registros']} init_q={len(plan['initial_queries'])}")

    mem = InvestigationMemory()
    graph = EvidenceGraph()

    for h in plan.get("hypotheses", []):
        if isinstance(h, dict) and h.get("topic"):
            mem.add_hypothesis(h["topic"], h.get("note",""))

    # período fiscal (1 vez)
    try:
        per_docs = hybrid_search(project_id, "0000 abertura período DT_INI DT_FIN CNPJ", k=6, source_kind="sped")
        periodo = extract_periodo_from_docs(per_docs)
    except Exception:
        periodo = None

    # ---------- FASE 2: INVESTIGATOR LOOP ----------
    pending_queries: List[Tuple[str,Optional[str],str]] = []  # (query, registro, topic)
    # bootstrap com initial_queries + ancoradas em registros do plano
    for q in plan.get("initial_queries", [])[:AGENT_SEARCHES_PER_ROUND*2]:
        pending_queries.append((q, None, q))
    for reg in plan.get("registros", [])[:6]:
        pending_queries.append((f"{reg} apuração valores totais", reg, f"reg_{reg}"))

    total_searches = 0
    for round_idx in range(AGENT_MAX_ROUNDS):
        if not pending_queries: break
        if total_searches >= AGENT_MAX_SEARCHES_TOTAL: break

        # seleciona até N queries não-repetidas
        batch: List[Tuple[str,Optional[str],str]] = []
        while pending_queries and len(batch) < AGENT_SEARCHES_PER_ROUND:
            q, reg, topic = pending_queries.pop(0)
            if mem.has_query(q): continue
            if total_searches + len(batch) >= AGENT_MAX_SEARCHES_TOTAL: break
            batch.append((q, reg, topic))

        if not batch:
            # nada novo p/ buscar → vai para crítico decidir
            pass
        else:
            job_update(job_id, stage=f"agent_round_{round_idx+1}", progress=10 + round_idx*12)
            # busca paralela
            tasks = [(project_id, q, AGENT_DEFAULT_K, reg, topic) for (q,reg,topic) in batch]
            with ThreadPoolExecutor(max_workers=min(8, len(tasks))) as ex:
                results = list(ex.map(_run_search_task, tasks))

            round_added = 0
            for (q, docs, topic) in results:
                mem.add_query(q); total_searches += 1
                # score interno = rank-based
                for i, d in enumerate(docs):
                    score = max(0.15,1.0 - (i * 0.07))
                    if mem.add_evidence(d, topic=topic, score=score): round_added += 1
                mem.mark_topic(topic)
            trace.append({"phase":"investigate","round":round_idx+1,"searches":len(batch),
                          "new_evidence":round_added,"total_evidence":len(mem.evidence)})
            print(f"[ORCH][{job_id}] round {round_idx+1} | searches={len(batch)} +ev={round_added} totev={len(mem.evidence)}")

        # atualiza grafo após cada rodada
        graph.ingest(mem)

        # ----- REFLECTION / CRITIC -----
        job_update(job_id, stage=f"agent_critic_{round_idx+1}")
        critique = critic_agent(plan, mem, graph, model=model_critic)
        u = critique.pop("_usage", {}); tokens_total["prompt"]+=u.get("prompt_tokens",0); tokens_total["completion"]+=u.get("completion_tokens",0)
        trace.append({"phase":"critic","round":round_idx+1,"ready":critique["ready"],
                      "gaps":critique["gaps"][:5],"reasoning":critique["reasoning"]})
        for g in critique["gaps"]:
            if g and g not in mem.gaps: mem.gaps.append(g)

        if critique["ready"] and len(mem.evidence) >= 5 and mem.coverage_score(plan.get("topics", [])) >= AGENT_REFLECTION_THRESHOLD:
            print(f"[ORCH][{job_id}] CRITIC ready=True após round {round_idx+1}")
            break

        # alimenta próxima rodada com refinamentos
        for q in critique.get("refined_queries", [])[:AGENT_SEARCHES_PER_ROUND]:
            if q and not mem.has_query(q): pending_queries.append((q, None, q))
        for r in critique.get("target_registros", [])[:3]:
            pending_queries.append((f"{r} valores apuração detalhamento", r, f"reg_{r}"))

        # sugere registros conectados ainda não cobertos (grafo)
        for r in graph.suggest_next_registros(mem, top=2):
            pending_queries.append((f"{r} ajustes débitos créditos", r, f"graph_{r}"))

    # ---------- Driva (opcional, 1 chamada compacta) ----------
    if plan.get("needs_driva"):
        try:
            d_docs = hybrid_search(
                project_id,
                "porte regime tributário CNAE sócios atividade",
                k=5,
                source_kind="driva"
            )
            for i, d in enumerate(d_docs):
                mem.add_evidence(d, topic="contexto_empresa", score=0.4 - i*0.02)

        except Exception as e:
            print(f"[ORCH][{job_id}] driva opcional falhou: {e}")

    # Analise deterministica monolitica: calculos, scoring, juridico e BI antes do LLM.
    analytics = build_analytics(mem)
    structured_analysis = build_final_analysis_schema(mem, analytics, graph, periodo)
    analytics["structured_analysis"] = structured_analysis

    trace.append({
        "phase": "analytics",
        "metrics": analytics,
        "structured_analysis_ready": True,
    })
    
    # ---------- FASE 3: SYNTH ----------
    job_update(job_id, stage="agent_synth", progress=90)
    final_md, synth_usage = synthesizer_agent(template, periodo, mem, analytics, graph, model=model_synth)
    tokens_total["prompt"]+=synth_usage.get("prompt_tokens",0)
    tokens_total["completion"]+=synth_usage.get("completion_tokens",0)
    trace.append({"phase":"synth","prompt_tokens":synth_usage.get("prompt_tokens",0),
                  "completion_tokens":synth_usage.get("completion_tokens",0)})

    elapsed_ms = int((time.time()-t0)*1000)
    return {
        "markdown":     final_md,
        "periodo":      periodo,
        "analytics": analytics,
        "structured_analysis": structured_analysis,
        "tokens": {
            "prompt":     tokens_total["prompt"],
            "completion": tokens_total["completion"],
            "total":      tokens_total["prompt"]+tokens_total["completion"],
        },
        "elapsed_ms":   elapsed_ms,
        "plan":         {k:v for k,v in plan.items() if not k.startswith("_")},
        "memory":       mem.snapshot(),
        "graph":        graph.export(),
        "searches":     total_searches,
        "model":        model_synth,
        "trace":        trace,
    }


# ==============================================================================
# PARSE MARKDOWN → STRUCTURED  (preservado)
# ==============================================================================
RE_CALC_LINE = re.compile(r"([A-Z][^=:]{3,60})\s*[:=]\s*(R\$\s?[\d\.,]+|\d[\d\.,]*)\s*(.*)")
RE_EVIDENCE_PIPE = re.compile(r"\|\s*([0-9A-Z]{4})\s*\|[^|\n]{2,200}\|")
RE_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[\.\)])\s+(.{15,})$", re.MULTILINE)

def _split_markdown_sections(md: str) -> Dict[str,str]:
    sections={}; cur="_intro"; buf=[]
    if not md: return sections
    for ln in md.splitlines():
        m = re.match(r"^\s*#{1,6}\s+(.+?)\s*$", ln)
        if m:
            sections[cur] = "\n".join(buf).strip()
            cur = re.sub(r"[^a-z0-9]+","_", m.group(1).lower()).strip("_")
            buf=[]
        else: buf.append(ln)
    sections[cur]="\n".join(buf).strip()
    return sections

def parse_summary_markdown(md: str) -> Dict[str, List[Any]]:
    out={"insights":[],"calculations":[],"data_crossings":[],"source_references":[]}
    if not md: return out
    sec = _split_markdown_sections(md); seen=set()
    for title, body in sec.items():
        if not any(kw in title for kw in ("sumar","exec","recomend","oportun","intelig","achad","risco")): continue
        for m in RE_BULLET.finditer(body):
            t = re.sub(r"\*\*","", m.group(1).strip())
            if "dado não disponível" in t.lower(): continue
            if t.lower() in seen: continue
            seen.add(t.lower())
            if len(out["insights"])<40: out["insights"].append(t)
    seen=set()
    for m in RE_CALC_LINE.finditer(md):
        d = m.group(1).strip(" -*").strip(); v=m.group(2).strip()
        if "dado não disponível" in (d+v).lower(): continue
        k=f"{d}|{v}"
        if k in seen: continue
        seen.add(k); out["calculations"].append({"description":d,"value":v,"formula":""})
        if len(out["calculations"])>=30: break
    seen=set()
    for m in RE_EVIDENCE_PIPE.finditer(md):
        full=m.group(0).strip(); reg=m.group(1).strip()
        if full in seen: continue
        seen.add(full)
        out["source_references"].append({"registro":reg,"excerpt":full[:400],"relevance":"evidência SPED"})
        if len(out["source_references"])>=40: break
    for p in re.split(r"\n\s*\n", md):
        regs = set(re.findall(r"\b([A-Z]\d{3})\b", p)) | set(re.findall(r"\|([0-9A-Z]{4})\|", p))
        if len(regs)>=2 and len(p)<1500:
            short=p.strip().replace("\n"," ")
            out["data_crossings"].append({"description":short[:600],"sources":sorted(regs),"result":""})
            if len(out["data_crossings"])>=20: break
    return out


# ==============================================================================
# MODELS
# ==============================================================================
class SummaryRequest(BaseModel):
    template: str
    query: Optional[str] = "gerar sumário geral"
    enrichment: Optional[Dict] = None  # DEPRECATED
    k: Optional[int] = 20
    project_id: str
    agentic: Optional[bool] = True
    model: Optional[str] = None  

class EnrichmentRequest(BaseModel):
    project_id: str
    enrichment: Dict
    source: Optional[str] = "manual_enrichment"

class ProcessPathsRequest(BaseModel):
    project_id: str
    paths: List[str]


# ==============================================================================
# INDEXAÇÃO (preservada)
# ==============================================================================
def process_job(job_id: str, files_data: List[dict], project_id: str):
    try:
        t0=time.time()
        job_update(job_id, status="processing", stage="chunking", progress=5)
        vs = get_vector_store(project_id)
        all_chunks, all_meta = [], []
        local_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        with ProcessPoolExecutor() as ex:
            texts=[f["text"] for f in files_data]
            results=list(ex.map(local_splitter.split_text, texts))
        for i, chunks in enumerate(results):
            fn = files_data[i]["filename"]
            for idx, ch in enumerate(chunks):
                reg = primary_registro(ch); regs_all = detect_registros(ch)
                all_chunks.append(ch)
                all_meta.append({"source":fn,"source_kind":"sped","chunk_index":idx,
                                 "project_id":project_id,"type":"document",
                                 "registro":reg or "unknown",
                                 "registros_all":",".join(regs_all) if regs_all else ""})
        n=len(all_chunks)
        print(f"[JOB {job_id}] chunking ok: {n} chunks em {time.time()-t0:.1f}s")
        job_update(job_id, progress=30, stage="embedding")
        IB=512; nb=(n+IB-1)//IB
        for b in range(nb):
            s=b*IB; e=min(s+IB,n)
            for at in range(3):
                try: vs.add_texts(texts=all_chunks[s:e], metadatas=all_meta[s:e]); break
                except Exception as ex_:
                    if "429" in str(ex_) and at<2: time.sleep(2*(at+1)); continue
                    raise
            job_update(job_id, progress=30+int(((b+1)/nb)*70))
        elapsed=time.time()-t0
        job_update(job_id, status="completed", stage="done", progress=100,
                   result={"total_chunks":n,"total_files":len(files_data),"elapsed_seconds":round(elapsed,1)})
        print(f"[JOB {job_id}] ✅ DONE em {elapsed:.1f}s")
    except Exception as e:
        print(traceback.format_exc()); job_update(job_id, status="error", error=str(e))


# ==============================================================================
# PROCESS SUMMARY  →  usa o novo orchestrator (com fallback one-shot)
# ==============================================================================
def _legacy_oneshot_summary(req: SummaryRequest, job_id: str, t_start: float):
    job_update(job_id, stage="legacy_oneshot")

    model_synth = resolve_model(req.model, LLM_MODEL_SYNTH)

    docs = hybrid_search(
        req.project_id,
        req.query or "sumário geral",
        k=min(req.k or 12, AGENT_MAX_K),
        source_kind="sped",
    )

    ctx_parts = []
    for d in docs:
        reg = d.metadata.get("registro") or primary_registro(d.page_content) or "?"
        ctx_parts.append(
            f"[{d.metadata.get('source')} | reg {reg}]\n"
            f"{d.page_content[:1500]}"
        )

    ctx = "\n\n---\n\n".join(ctx_parts)[:12000]
    periodo = extract_periodo_from_docs(docs)

    prompt = f"""Você é um auditor fiscal SPED.
PERÍODO: {periodo or '?'}

CONTEXTO:
{ctx}

TEMPLATE:
{req.template}

REGRAS:
- Não invente.
- Cite arquivo + registro.
- Se faltar evidência, omita o item.
"""

    llm = ChatOpenAI(model=model_synth, temperature=0.0, api_key=OPENAI_API_KEY)

    with get_openai_callback() as cb:
        resp = llm.invoke(prompt)
        clean_markdown = _strip_md_fences(resp.content)
        tok = cb.total_tokens
        pt = cb.prompt_tokens
        ct = cb.completion_tokens

    structured = parse_summary_markdown(clean_markdown) or {}

    final_content = {
        "visao_geral": structured.get("overview") or clean_markdown,
        "insights": structured.get("insights") or [],
        "inconsistencias": structured.get("inconsistencies") or [],
        "oportunidades": structured.get("opportunities") or [],
        "analises": structured.get("analyses") or [],
        "calculos": structured.get("calculations") or {},
        "cruzamento_de_dados": structured.get("data_crossings") or {},
        "justificativas": structured.get("justifications") or [],
        "referencias": structured.get("source_references") or [],
    }

    final_content = {
        k: v for k, v in final_content.items()
        if v not in (None, "", [], {})
    }

    if not final_content:
        final_content = {"texto": clean_markdown}

    job_update(
        job_id,
        status="completed",
        stage="done",
        progress=100,
        result={
            "summary": final_content,
            "mode": "legacy_oneshot",
            "model": model_synth,
            "model_used": model_synth,
            "tokens_used": tok,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "generation_time_ms": int((time.time() - t_start) * 1000),
            "periodo_detectado": periodo,
        },
    )


def _strip_md_fences(text: str) -> str:
    """
    Remove cercas markdown ```markdown ... ```
    quando o modelo embrulha TODA a resposta em code block.
    """

    if not isinstance(text, str):
        return text

    t = text.strip()

    # remove bloco completo ```lang ... ```
    m = re.match(
        r"^```(?:markdown|md|text)?\s*\n(.*)\n```$",
        t,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if m:
        return m.group(1).strip()

    # fallback
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```$", "", t)

    return t.strip()

def process_summary_job(job_id: str, req: SummaryRequest):
    t0 = time.time()

    try:
        job_update(job_id, status="processing", stage="starting")
        print(f"[SUMMARY][{job_id}] 🚀 START | agentic={req.agentic}")

        if req.enrichment:
            print(f"[SUMMARY][{job_id}] ⚠️ enrichment in-body DEPRECATED; use /enrichment")

        if req.agentic:
            try:
                out = orchestrate_summary(req, job_id)
                raw_markdown = _strip_md_fences(out.get("markdown", "") or "")
                print(f"[SUMMARY][{job_id}] RAW_MARKDOWN_EMPTY => {not bool(raw_markdown.strip())}")

                print(f"[SUMMARY][{job_id}] RAW_MD_LEN => {len(raw_markdown)}")

                structured = parse_summary_markdown(raw_markdown) or {}

                final_content = {
                 # ⭐ PRINCIPAL
                 "visao_geral": raw_markdown,

                 # auxiliares
                 "insights": structured.get("insights") or [],
                 "calculos": structured.get("calculations") or [],
                 "cruzamento_de_dados": structured.get("data_crossings") or [],
                 "referencias": structured.get("source_references") or [],
                }

                final_content = {
                    k: v for k, v in final_content.items()
                    if v not in (None, "", [], {})
                }
                print(f"[SUMMARY][{job_id}] FINAL_CONTENT_KEYS => {list(final_content.keys())}")

                tokens = out.get("tokens") or {}

                job_update(
                    job_id,
                    status="completed",
                    stage="done",
                    progress=100,
                    result={
                        # O frontend salva SOMENTE rawResult.summary em summaries.content
                        "summary": final_content,

                        # Metadados ficam fora do JSON visual do sumário
                        "mode": "agentic_v2",
                        "analytics": out.get("analytics"),
                        "structured_analysis": out.get("structured_analysis"),
                        "model": out.get("model"),
                        "model_used": out.get("model"),
                        "tokens_used": tokens.get("total"),
                        "prompt_tokens": tokens.get("prompt"),
                        "completion_tokens": tokens.get("completion"),
                        "generation_time_ms": out.get("elapsed_ms"),
                        "periodo_detectado": out.get("periodo"),
                    },
                )

                print(
                    f"[SUMMARY][{job_id}] ✅ AGENTIC_V2 DONE | "
                    f"searches={out.get('searches')} | "
                    f"tokens={tokens.get('total')} | {out.get('elapsed_ms')}ms"
                )
                return

            except Exception as e:
                print(f"[SUMMARY][{job_id}] ⚠️ orchestrator falhou: {e}")
                print(traceback.format_exc())

                job_update(
                job_id,
                status="error",
                stage="agentic_failed",
                error=str(e)
                )
                # fallback para one-shot simples
                _legacy_oneshot_summary(req, job_id, t0)

        

    except Exception as e:
        print(traceback.format_exc())
        job_update(job_id, status="error", error=str(e))

# ==============================================================================
# ENRICHMENT (Driva) — preservado
# ==============================================================================
def enrichment_to_text(data, parent_key=""):
    texts=[]
    if isinstance(data, dict):
        for k,v in data.items():
            nk = f"{parent_key}.{k}" if parent_key else k
            texts.append(enrichment_to_text(v, nk))
    elif isinstance(data, list):
        for i,it in enumerate(data):
            texts.append(enrichment_to_text(it, f"{parent_key}[{i}]"))
    else:
        texts.append(f"{parent_key}: {data}")
    return "\n".join(texts)


@app.post("/enrichment")
async def upload_enrichment(req: EnrichmentRequest):
    try:
        vs = get_vector_store(req.project_id)
        f = preprocess_driva(req.enrichment)
        if not f:
            return {"status":"success","chunks_saved":0,"message":"Nenhum campo Driva relevante"}
        txt = enrichment_to_text(f)
        chunks = text_splitter.split_text(txt)
        texts, metas = [], []
        for idx, ch in enumerate(chunks):
            texts.append(ch)
            metas.append({"project_id":req.project_id,"type":"enrichment","source_kind":"driva",
                          "source":req.source,"chunk_index":idx,"registro":"enrichment"})
        vs.add_texts(texts=texts, metadatas=metas)
        return {"status":"success","chunks_saved":len(chunks),
                "fields_kept":list(f.keys()) if isinstance(f,dict) else []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# UPLOAD / PROCESS-PATHS / DELETE / STATUS  — preservados
# ==============================================================================
@app.post("/upload")
async def upload_documents(project_id: str, files: List[UploadFile] = File(...)):
    try:
        if not project_id: raise HTTPException(status_code=400, detail="project_id obrigatório")
        job_id = str(uuid.uuid4())
        async def proc(file: UploadFile):
            content = await file.read(); text = smart_decode_sped(content)
            return {"filename":file.filename,"text":text} if text.strip() else None
        results = await asyncio.gather(*[proc(f) for f in files])
        files_data = [r for r in results if r]
        if not files_data: raise HTTPException(status_code=400, detail="Nenhum arquivo válido")
        job_create(job_id, kind="process", project_id=project_id)
        threading.Thread(target=process_job, args=(job_id, files_data, project_id)).start()
        return {"job_id":job_id, "project_id":project_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _download_file(path: str) -> Optional[dict]:
    try:
        data = supabase.storage.from_(BUCKET_NAME).download(path)
        text = smart_decode_sped(data)
        if not text.strip(): return None
        return {"filename": path.split("/")[-1], "text": text}
    except Exception as e:
        print(f"[download] falhou {path}: {e}"); return None


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
                t0=time.time()
                with ThreadPoolExecutor(max_workers=16) as ex:
                    results = list(ex.map(_download_file, req.paths))
                files_data = [r for r in results if r]
                print(f"[JOB {job_id}] download {len(files_data)}/{len(req.paths)} em {time.time()-t0:.1f}s")
                if not files_data:
                    job_update(job_id, status="error", error="Nenhum arquivo válido baixado"); return
                process_job(job_id, files_data, req.project_id)
            except Exception as e:
                print(traceback.format_exc()); job_update(job_id, status="error", error=str(e))
        threading.Thread(target=runner).start()
        return {"job_id":job_id, "project_id":req.project_id, "total_files":len(req.paths)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-project/{project_id}")
def delete_project(project_id: str, folder_path: str):
    try:
        if not project_id or not folder_path:
            raise HTTPException(status_code=400, detail="project_id e folder_path obrigatórios")
        try:
            while True:
                files = supabase.storage.from_(BUCKET_NAME).list(path=folder_path)
                if not files: break
                paths = [f"{folder_path.rstrip('/')}/{f['name']}" for f in files]
                supabase.storage.from_(BUCKET_NAME).remove(paths)
                time.sleep(0.2)
        except Exception as e:
            print("⚠️ storage:", e)
        p = os.path.join(PERSIST_DIR, project_id)
        if os.path.exists(p): shutil.rmtree(p)
        return {"status":"success","message":f"Project {project_id} deletado"}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}")
def get_status(job_id: str):
    j = job_get(job_id)
    if not j: raise HTTPException(status_code=404, detail="Job não encontrado")
    return j

@app.get("/summary-status/{job_id}")
def get_summary_status(job_id: str):
    j = job_get(job_id)
    if not j: raise HTTPException(status_code=404, detail="Summary job não encontrado")
    return j

@app.post("/generate-summary")
async def generate_summary(req: SummaryRequest):
    try:
        job_id = str(uuid.uuid4())
        job_create(job_id, kind="summary", project_id=req.project_id)
        threading.Thread(target=process_summary_job, args=(job_id, req)).start()
        return {"job_id": job_id, "status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
