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

app = FastAPI()

from fastapi.responses import Response

@app.middleware("http")
async def add_cors_headers(request, call_next):
    response: Response = await call_next(request)

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"

    return response


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

# -------------------------------
# RAG
# -------------------------------
def get_context(query: str, project_id: str, k: int = 10):
    try:
        vector_store = get_vector_store(project_id)
        docs = vector_store.similarity_search(query, k=k)

        if not docs:
            return "Nenhum dado encontrado para este projeto."

        context = "\n\n".join([
            f"[Fonte: {doc.metadata.get('source')} | Chunk: {doc.metadata.get('chunk_index')}]\n{doc.page_content}"
            for doc in docs
        ])

        return context

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao buscar contexto: {str(e)}")

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
FORMATO DE RESPOSTA (JSON OBRIGATÓRIO)
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
    k: Optional[int] = 5
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

        job["stage"] = "chunking"

        for i, file in enumerate(files_data):
            chunks = text_splitter.split_text(file["text"])

            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": file["filename"],
                    "chunk_index": idx,
                    "project_id": project_id
                })

            job["progress"] = int((i + 1) / len(files_data) * 50)

        job["stage"] = "embedding"

        BATCH_SIZE = 100

        for i in range(0, len(all_chunks), BATCH_SIZE):
            vector_store.add_texts(
                texts=all_chunks[i:i+BATCH_SIZE],
                metadatas=all_metadata[i:i+BATCH_SIZE]
            )

            job["progress"] = 50 + int((i / len(all_chunks)) * 50)

        job["status"] = "completed"
        job["stage"] = "done"
        job["progress"] = 100

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
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
        files_data = []

        for file in files:
            content = await file.read()
            text = content.decode("utf-8", errors="ignore")

            if text.strip():
                files_data.append({
                    "filename": file.filename,
                    "text": text
                })

        if not files_data:
            raise HTTPException(status_code=400, detail="Nenhum arquivo válido")

        jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "stage": "upload",
            "project_id": project_id
        }

        threading.Thread(
            target=process_job,
            args=(job_id, files_data, project_id)
        ).start()

        return {"job_id": job_id, "project_id": project_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# STATUS
# -------------------------------



@app.delete("/delete-project/{project_id}")
def delete_project(project_id: str):
    try:
        if not project_id:
            raise HTTPException(status_code=400, detail="project_id obrigatório")

        project_path = os.path.join(PERSIST_DIR, project_id)

        # 🔥 DELETE TOTAL DIRETO
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
        else:
            raise HTTPException(
                status_code=404,
                detail="Projeto não encontrado"
            )

        return {
            "status": "success",
            "message": f"Projeto {project_id} deletado COMPLETAMENTE"
        }

    except HTTPException:
        raise

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
        context = get_context(req.query, req.project_id, req.k)

        # 🔥 LOGS IMPORTANTES
        print("===================================")
        print("📊 DEBUG TAMANHO")
        print("Project:", req.project_id)
        print("Query:", req.query)
        print("Chunks solicitados (k):", req.k)
        print("Context size:", len(context))
        print("Enrichment size:", len(str(req.enrichment)) if req.enrichment else 0)

        # 🔥 LIMITE HARD
        context = context[:12000]

        prompt = build_prompt(req.template, context, req.enrichment)

        print("Prompt size:", len(prompt))
        print("===================================")

        response = llm.invoke(prompt)

        return {
            "summary": response.content,
            "project_id": req.project_id
        }

    except Exception as e:
        print("🔥 ERRO NO SUMMARY")
        print(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar resumo: {str(e)}"
        )