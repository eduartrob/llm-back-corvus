import json
import logging
import asyncio
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from pydantic import BaseModel

from app.api.models import (
    AnalyzeProposalRequest,
    StartSessionRequest,
    StartSessionResponse,
    SessionMessageRequest,
    SessionMessageResponse,
    GenerateNameRequest,
    GenerateRAGSummaryRequest,
    AnalyzeHomeworkRequest,
    FilterSoftwareRequest,
    FilterSoftwareResponse
)
from app.services.ollama_client import ollama_client
from app.services.session_store import session_store
from app.services.quota_client import quota_client
from app.services.llm_queue import llm_queue

router = APIRouter()
logger = logging.getLogger(__name__)

class BlueOceanRequest(BaseModel):
    title: str
    description: str
    category: str

# prompt de analisis de propuesta reutilizado del clustering service

from app.core.prompts import (
    BLUE_OCEAN_SYSTEM_PROMPT,
    build_blue_ocean_user_prompt,
    build_groq_analysis_prompt,
    build_ollama_analysis_prompt,
    build_rag_summary_prompt
)

@router.get("/health")
async def health():
    ollama_ok = ollama_client.check_health()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "service": "llm-back-corvus",
        "ollama": "connected" if ollama_ok else "unreachable",
        "model": ollama_client.model,
    }

@router.post("/analyze-blue-ocean")
async def analyze_blue_ocean(body: BlueOceanRequest):
    async def _do_analysis():
        system_prompt = BLUE_OCEAN_SYSTEM_PROMPT
        user_prompt = build_blue_ocean_user_prompt(body.title, body.description, body.category)
        
        # Intentar con Groq primero
        try:
            from app.api.groq_client import analyze_with_groq
            logger.info("[analyze-blue-ocean] Intentando usar GroqCloud...")
            # analyze_with_groq es síncrona, ejecutamos en hilo
            result = await asyncio.to_thread(analyze_with_groq, system_prompt, user_prompt)
            return result
        except Exception as e:
            logger.warning(f"[analyze-blue-ocean] Falló GroqCloud ({e}). Haciendo failover a Ollama local...")
            
        # Fallback a Ollama
        if not ollama_client.check_health():
            raise HTTPException(status_code=503, detail="El motor de IA (Ollama) no está disponible.")

        try:
            raw_response = await ollama_client.generate(prompt=user_prompt, system_prompt=system_prompt)
            
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
                
            return json.loads(cleaned_response)
        except Exception as e:
            logger.error(f"[analyze-blue-ocean] Error con Ollama: {e}")
            raise HTTPException(status_code=503, detail="El motor de IA (Ollama) falló al procesar la solicitud.")

    # Prioridad baja (10) para tareas de fondo
    return await llm_queue.enqueue(10, _do_analysis())



@router.post("/analyze-proposal")
async def analyze_proposal(body: AnalyzeProposalRequest):
    async def _do_analysis():
        context_text = ""
        for i, proj in enumerate(body.similar_projects):
            sim_pct = proj.get("similarity_pct", 0)
            context_text += (
                f"\n--- Proyecto Existente {i+1} ---\n"
                f"Título: {proj.get('title', 'Desconocido')}\n"
                f"Similitud Matemática (ChromaDB): {sim_pct:.1f}%\n"
                f"Contenido: {proj.get('content', '')}\n"
            )

        groq_system, groq_user = build_groq_analysis_prompt(
            proposal_text=body.proposal_text,
            context_text=context_text,
            project_name=body.project_name,
            top_project_name=body.top_project_name,
            max_sim_pct=body.max_sim_pct,
            risk_level=body.risk_level,
        )

        if body.provider == "groq":
            from app.api.groq_client import analyze_with_groq
            try:
                logger.info("[analyze-proposal] Intentando usar GroqCloud...")
                result = await asyncio.to_thread(analyze_with_groq, groq_system, groq_user)
                return result
            except Exception as e:
                logger.warning(f"[analyze-proposal] Falló GroqCloud ({e}). Haciendo failover a Ollama local...")

        # Flujo normal o fallback a Ollama
        if not ollama_client.check_health():
            raise HTTPException(status_code=503, detail="El motor de IA (Ollama) no está disponible.")

        ollama_system, ollama_user = build_ollama_analysis_prompt(
            proposal_text=body.proposal_text,
            context_text=context_text,
            project_name=body.project_name,
            top_project_name=body.top_project_name,
            max_sim_pct=body.max_sim_pct,
            risk_level=body.risk_level,
        )

        try:
            raw_response = await ollama_client.generate(
                prompt=ollama_user,
                system_prompt=ollama_system
            )
            
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
                
            result = json.loads(cleaned_response)
            return result
        except json.JSONDecodeError:
            logger.error(f"[analyze-proposal] Ollama no devolvió JSON válido: {raw_response}")
            raise HTTPException(status_code=500, detail="El modelo no devolvió un formato válido.")
        except Exception as e:
            logger.error(f"[analyze-proposal] Error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Prioridad alta (1) para interacciones de usuario
    return await llm_queue.enqueue(1, _do_analysis())

@router.post("/session/start", response_model=StartSessionResponse)
async def start_session(
    body: StartSessionRequest,
    x_user_data: Optional[str] = Header(None),
):
    
    quota = quota_client.get_quota(body.user_id)
    if not quota["can_create"]:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "quota_exceeded",
                "message": f"Has usado tus {quota['limit']} sesiones gratuitas de análisis IA. Actualiza tu plan para continuar.",
                "sessions_used": quota["sessions_used"],
                "sessions_limit": quota["limit"],
            },
        )

    if not ollama_client.check_health():
        raise HTTPException(status_code=503, detail="El motor de IA no está disponible en este momento.")

    approved = body.analysis_result.get("approved", False)
    mode = "defense" if approved else "rejection"

    session = session_store.create(
        user_id=body.user_id,
        mode=mode,
        analysis_result=body.analysis_result,
        proposal_summary=body.proposal_summary,
    )

    quota_client.register_session(
        user_id=body.user_id,
        session_data={
            "session_id": session.session_id,
            "verdict": "approved" if approved else "rejected",
            "proposal_summary": body.proposal_summary[:500],
            "analysis_json": body.analysis_result,
        },
    )

    if mode == "defense":
        opening_prompt = (
            "El alumno acaba de ver que su proyecto fue aprobado. "
            "Abre la sesión de defensa presentándote brevemente y lanzando tu primera pregunta difícil "
            "sobre el punto más débil que identificas en el proyecto. Sé directo y específico."
        )
    else:
        score = body.analysis_result.get("innovation_index", {}).get("score", 0)
        risk = body.analysis_result.get("semantic_collision_risk", {}).get("alert_type", "")
        opening_prompt = (
            f"El alumno acaba de recibir el rechazo de su propuesta (score: {score}%, riesgo: {risk}). "
            "Abre la sesión presentándote como asesor constructivo, explica brevemente las razones principales del rechazo "
            "y pregúntale qué parte le gustaría entender mejor primero."
        )

    messages = session.to_ollama_messages()
    messages.append({"role": "user", "content": opening_prompt})

    async def _do_start_chat():
        try:
            ai_opening = await ollama_client.chat(messages, temperature=0.6)
            session.add_message("assistant", ai_opening)
            return ai_opening
        except Exception as e:
            logger.error(f"[session/start] Error generando apertura: {e}")
            raise HTTPException(status_code=500, detail="Error generando el mensaje inicial de la IA.")

    ai_opening = await llm_queue.enqueue(1, _do_start_chat())

    return StartSessionResponse(
        session_id=session.session_id,
        mode=mode,
        ai_opening_message=ai_opening,
        quota=quota,
    )

@router.post("/session/message", response_model=SessionMessageResponse)
async def session_message(body: SessionMessageRequest):
    
    session = session_store.get(body.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Sesión no encontrada o expirada (las sesiones expiran tras 30 minutos de inactividad).",
        )

    if not ollama_client.check_health():
        raise HTTPException(status_code=503, detail="El motor de IA no está disponible en este momento.")

    session.add_message("user", body.user_message)

    full_messages = session.to_ollama_messages()

    async def _do_chat():
        try:
            ai_response = await ollama_client.chat(full_messages, temperature=0.6)
            session.add_message("assistant", ai_response)
            return ai_response
        except Exception as e:
            logger.error(f"[session/message] Error: {e}")
            session.messages.pop()
            raise HTTPException(status_code=500, detail="Error procesando tu mensaje. Intenta de nuevo.")

    ai_response = await llm_queue.enqueue(1, _do_chat())

    return SessionMessageResponse(
        ai_message=ai_response,
        session_id=body.session_id,
    )

@router.post("/generate-name")
async def generate_name(body: GenerateNameRequest):
    """
    Genera un nombre de máximo 2 palabras para un clúster de proyectos.
    Usa el proveedor de IA configurado en el sistema (Groq o Ollama).
    """
    system_prompt = (
        "Eres un experto en clasificación de proyectos académicos de ingeniería de software. "
        "Analiza los fragmentos de proyectos que te dan y responde ÚNICAMENTE con un nombre "
        "de exactamente 2 palabras en español que describa su área temática principal. "
        "No uses comillas, no des explicaciones, no escribas nada más. Solo las 2 palabras."
    )

    async def _do_generate():
        raw_response = None

        # Intentar con el proveedor configurado
        if body.provider == "groq":
            try:
                from app.api.groq_client import generate_text_with_groq
                logger.info("[generate-name] Intentando con Groq...")
                raw_response = await asyncio.to_thread(
                    generate_text_with_groq, system_prompt, body.prompt
                )
            except Exception as e:
                logger.warning(f"[generate-name] Groq falló ({e}). Failover a Ollama...")

        # Fallback o flujo directo a Ollama
        if raw_response is None:
            if not ollama_client.check_health():
                raise HTTPException(status_code=503, detail="El motor de IA (Ollama) no está disponible.")
            try:
                raw_response = await ollama_client.generate(
                    prompt=body.prompt,
                    system_prompt=system_prompt
                )
            except Exception as e:
                logger.error(f"[generate-name] Error con Ollama: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Limpiar y limitar a 2 palabras
        cleaned = raw_response.strip().strip('"\'.,: \n')
        words = cleaned.split()
        name = " ".join(words[:2]) if words else "Tema Tecnológico"
        logger.info(f"[generate-name] Nombre generado: '{name}'")
        return {"name": name}

    return await llm_queue.enqueue(10, _do_generate())

@router.post("/generate-rag-summary")
async def generate_rag_summary(body: GenerateRAGSummaryRequest):
    """
    Genera un resumen usando Llama3 a través de Groq o Ollama,
    basándose únicamente en el contexto recuperado de ChromaDB.
    """
    if not body.context.strip():
        return {"summary": "Lo siento, no encontré información sobre este tema en los apuntes de la clase."}

    system_prompt, user_prompt = build_rag_summary_prompt(body.query, body.context)

    async def _do_generate():
        raw_response = None

        if body.provider == "groq":
            try:
                from app.api.groq_client import generate_text_with_groq
                logger.info("[generate-rag-summary] Intentando con Groq...")
                raw_response = await asyncio.to_thread(
                    generate_text_with_groq, system_prompt, user_prompt
                )
            except Exception as e:
                logger.warning(f"[generate-rag-summary] Groq falló ({e}). Failover a Ollama...")

        if raw_response is None:
            if not ollama_client.check_health():
                resumen_simulado = body.context[:500].strip()
                if len(body.context) > 500:
                    resumen_simulado += "..."
                return {"summary": f"Hubo un error contactando a la IA, pero aquí está el extracto más relevante:\n\n❝ {resumen_simulado} ❞"}
            
            try:
                raw_response = await ollama_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt
                )
            except Exception as e:
                logger.error(f"[generate-rag-summary] Error con Ollama: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        return {"summary": raw_response}

    # Prioridad alta para interacciones de chat
    return await llm_queue.enqueue(1, _do_generate())

@router.post("/analyze-homework")
async def analyze_homework(body: AnalyzeHomeworkRequest):
    """
    Analiza una tarea (documento de texto completo) para extraer tecnologías y detectar IA.
    """
    system_prompt = (
        "Eres un analizador técnico de tareas académicas. "
        "A partir del título y el texto completo proporcionado, tu objetivo es extraer una lista "
        "de las tecnologías (lenguajes, frameworks, herramientas de TI) principales que se mencionan. "
        "Además, debes analizar el texto y dar una puntuación sobre qué tan probable es que haya sido "
        "generado por Inteligencia Artificial (ChatGPT, Claude, etc).\n"
        "Debes responder ESTRICTAMENTE con un objeto JSON (sin comillas invertidas ni bloques ```) "
        "que tenga esta estructura exacta:\n"
        "{\n"
        "  \"tecnologias_detectadas\": [\n"
        "    {\"tecnologia\": \"Python\", \"score\": 0.95},\n"
        "    {\"tecnologia\": \"React\", \"score\": 0.80}\n"
        "  ],\n"
        "  \"es_ia\": false,\n"
        "  \"probabilidad_ia\": 0.15\n"
        "}\n"
        "El 'score' de tecnología es un float entre 0.0 y 1.0 indicando relevancia. "
        "El campo 'es_ia' es booleano (true si probabilidad_ia >= 0.75). 'probabilidad_ia' es float 0.0 a 1.0."
    )
    
    # Si el texto es absurdamente largo (>12000 caracteres), tomamos el principio y el final.
    text_to_process = body.full_text
    if len(text_to_process) > 12000:
        text_to_process = text_to_process[:6000] + "\n\n... [TEXTO OMITIDO] ...\n\n" + text_to_process[-6000:]
        
    user_prompt = f"Título de la Tarea: {body.title}\n\nContenido de la tarea:\n{text_to_process}"

    async def _do_analysis():
        raw_response = None

        if body.provider == "groq":
            try:
                from app.api.groq_client import analyze_with_groq
                logger.info("[analyze-homework] Intentando con Groq...")
                raw_response = await asyncio.to_thread(
                    analyze_with_groq, system_prompt, user_prompt
                )
            except Exception as e:
                logger.warning(f"[analyze-homework] Groq falló ({e}). Failover a Ollama...")

        if raw_response is None:
            if not ollama_client.check_health():
                raise HTTPException(status_code=503, detail="El motor de IA (Ollama) no está disponible.")
            try:
                raw_response = await ollama_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt
                )
            except Exception as e:
                logger.error(f"[analyze-homework] Error con Ollama: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        try:
            if isinstance(raw_response, dict):
                data = raw_response
            else:
                cleaned = raw_response.strip()
                if cleaned.startswith("```json"): cleaned = cleaned[7:]
                if cleaned.startswith("```"): cleaned = cleaned[3:]
                if cleaned.endswith("```"): cleaned = cleaned[:-3]
                
                data = json.loads(cleaned)
            # Asegurar formato
            if "tecnologias_detectadas" not in data:
                data["tecnologias_detectadas"] = []
            return data
        except json.JSONDecodeError:
            logger.error(f"[analyze-homework] La IA no devolvió un JSON válido: {raw_response}")
            return {"tecnologias_detectadas": [], "es_ia": None, "probabilidad_ia": None}

    # Prioridad media para tareas
    return await llm_queue.enqueue(5, _do_analysis())

@router.post("/filter-software-documents", response_model=FilterSoftwareResponse)
async def filter_software_documents(req: FilterSoftwareRequest):
    """
    Evalúa una lista de documentos por título y carpeta.
    Retorna únicamente los IDs que semánticamente corresponden a software,
    programación, sistemas, bases de datos o tecnología.
    """
    async def _do_filter():
        if not req.documents:
            return {"valid_ids": []}

        system_prompt = (
            "Eres un filtro experto de materias universitarias. Tu único propósito es evaluar una lista "
            "de documentos (tareas, materias, proyectos) y determinar cuáles pertenecen al área de "
            "Ingeniería de Software, Programación, Sistemas, Bases de Datos, Redes, Arquitectura o afines.\n"
            "IGNORA materias de tronco común, humanidades, inglés, valores, deportes, historia, derecho, etc.\n"
            "Responde ÚNICAMENTE con un JSON que tenga la clave 'valid_ids' conteniendo un arreglo "
            "de los IDs aprobados. No incluyas texto extra."
        )

        lista_docs = [{"id": d.id, "name": d.name, "folder": d.folder} for d in req.documents]
        user_prompt = f"Evalúa esta lista y devuelve los IDs válidos en JSON:\n{json.dumps(lista_docs, ensure_ascii=False)}"

        try:
            if not ollama_client.check_health():
                logger.warning("[filter-software] Ollama no disponible, aprobando todos por fallback.")
                return {"valid_ids": [d.id for d in req.documents]}

            raw_response = await ollama_client.generate(
                prompt=user_prompt,
                system_prompt=system_prompt
            )
            
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.startswith("```"): cleaned = cleaned[3:]
            if cleaned.endswith("```"): cleaned = cleaned[:-3]
            
            data = json.loads(cleaned)
            if "valid_ids" not in data or not isinstance(data["valid_ids"], list):
                return {"valid_ids": []}
            return data
        except Exception as e:
            logger.error(f"[filter-software] Error procesando con Ollama: {e}")
            return {"valid_ids": [d.id for d in req.documents]}

    return await llm_queue.enqueue(3, _do_filter())
