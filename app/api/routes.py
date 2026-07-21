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
    FilterSoftwareResponse,
    GenerateCareerSkillsRequest,
    GenerateCareerSkillsResponse,
    ValidateIdeaQuickRequest
)
from app.services.ollama_client import ollama_client
from app.services.session_store import session_store
from app.services.quota_client import quota_client
from app.services.llm_queue import llm_queue
from app.api.groq_client import chat_with_groq

router = APIRouter()
logger = logging.getLogger(__name__)

class BlueOceanRequest(BaseModel):
    title: str
    description: str
    category: str
    groq_model: Optional[str] = "llama-3.1-8b-instant"

# prompt de analisis de propuesta reutilizado del clustering service

from app.core.prompts import (
    BLUE_OCEAN_SYSTEM_PROMPT,
    build_blue_ocean_user_prompt,
    build_groq_analysis_prompt,
    build_ollama_analysis_prompt,
    build_rag_summary_prompt
)

def _enforce_approved_flag(result: dict) -> dict:
    if "approved" not in result or not isinstance(result["approved"], bool):
        score = result.get("innovation_index", {}).get("score", 0)
        quality = result.get("quality_metrics", {})
        academic = quality.get("academic_rigor", 0)
        technical = quality.get("technical_relevance", 0)
        
        avg_quality = (academic + technical) / 2
        
        result["approved"] = score >= 70 and avg_quality >= 70
    return result

@router.get("/health")
async def health():
    ollama_ok = ollama_client.check_health()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "service": "llm-back-corvus",
        "ollama": "connected" if ollama_ok else "unreachable",
        "model": ollama_client.model,
    }

@router.get("/groq-models")
async def get_groq_models():
    from app.api.groq_client import list_groq_models
    try:
        models = await asyncio.to_thread(list_groq_models)
        return {"status": "success", "data": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-blue-ocean")
async def analyze_blue_ocean(body: BlueOceanRequest):
    async def _do_analysis():
        system_prompt = BLUE_OCEAN_SYSTEM_PROMPT
        user_prompt = build_blue_ocean_user_prompt(body.title, body.description, body.category)
        
        # Intentar con Groq primero
        try:
            from app.api.groq_client import analyze_with_groq
            logger.info("[analyze-blue-ocean] Intentando usar GroqCloud con Llama 70B...")
            # analyze_with_groq es síncrona, ejecutamos en hilo
            result = await asyncio.to_thread(analyze_with_groq, system_prompt, user_prompt, "llama-3.3-70b-versatile")
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



# ── Almacén de diagnóstico para el último prompt enviado ─────────────────
_last_prompt_debug = {}

@router.post("/analyze-proposal")
async def analyze_proposal(body: AnalyzeProposalRequest):
    async def _do_analysis():
        context_text = ""
        global _last_prompt_debug
        for i, proj in enumerate(body.similar_projects):
            sim_pct = proj.get("similarity_pct", 0)
            content = proj.get('content', '')
            # Truncar contenido de similares a 400 chars para no saturar el prompt
            content_truncated = content[:400] + "..." if len(content) > 400 else content
            context_text += (
                f"\n--- Proyecto Existente {i+1} ---\n"
                f"Título: {proj.get('title', 'Desconocido')}\n"
                f"Similitud Matemática (Qdrant): {sim_pct:.1f}%\n"
                f"Fragmento donde se solapa: {content_truncated}\n"
            )
        
        logger.info(
            f"[analyze-proposal] Recibido: proposal_text={len(body.proposal_text)} chars, "
            f"similar_projects={len(body.similar_projects)}, "
            f"max_sim_pct={body.max_sim_pct}, risk_level={body.risk_level}, "
            f"provider={body.provider}"
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
                logger.info(
                    f"[analyze-proposal] Enviando a Groq: system_prompt={len(groq_system)} chars, "
                    f"user_prompt={len(groq_user)} chars, model={body.groq_model}"
                )
                # Guardar para diagnóstico
                _last_prompt_debug = {
                    "provider": "groq",
                    "model": body.groq_model,
                    "system_prompt_len": len(groq_system),
                    "user_prompt_len": len(groq_user),
                    "system_preview": groq_system[:300],
                    "user_preview": groq_user[:500],
                    "timestamp": str(__import__('datetime').datetime.now()),
                }
                result = await asyncio.to_thread(analyze_with_groq, groq_system, groq_user, body.groq_model)
                logger.info(f"[analyze-proposal] Groq respondió exitosamente con modelo: {result.get('actual_model_used', 'desconocido')}")
                return _enforce_approved_flag(result)
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
            logger.info(
                f"[analyze-proposal] Enviando a Ollama: system_prompt={len(ollama_system)} chars, "
                f"user_prompt={len(ollama_user)} chars"
            )
            # Guardar para diagnóstico
            _last_prompt_debug = {
                "provider": "ollama",
                "model": ollama_client.model,
                "system_prompt_len": len(ollama_system),
                "user_prompt_len": len(ollama_user),
                "system_preview": ollama_system[:300],
                "user_preview": ollama_user[:500],
                "timestamp": str(__import__('datetime').datetime.now()),
            }
            raw_response = await ollama_client.generate(
                prompt=ollama_user,
                system_prompt=ollama_system
            )
            logger.info(f"[analyze-proposal] Ollama respondió: {len(raw_response)} chars")
            
            cleaned_response = raw_response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
                
            result = json.loads(cleaned_response)
            return _enforce_approved_flag(result)
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
    
    # quota_client.get_quota(body.team_id) could be changed to team-based quota later.
    # For now, let's assume quota is handled at team level or bypass if needed.
    # Using team_id for quota registration:
    quota = quota_client.get_quota(body.team_id)
    if not quota["can_create"]:
        raise HTTPException(
            status_code=402,
            detail={
                "error": "quota_exceeded",
                "message": f"El equipo ha usado sus {quota['limit']} sesiones gratuitas de análisis IA.",
                "sessions_used": quota["sessions_used"],
                "sessions_limit": quota["limit"],
            },
        )

    if not ollama_client.check_health():
        raise HTTPException(status_code=503, detail="El motor de IA no está disponible en este momento.")

    actual_analysis = body.analysis_result.get("ollama_analysis", body.analysis_result)
    
    approved = actual_analysis.get("approved", False)
    mode = "defense" if approved else "rejection"

    existing_session = session_store.get_by_team_id(body.team_id)
    if existing_session:
        ai_opening = ""
        for m in existing_session.messages:
            if m["role"] == "assistant":
                ai_opening = m["content"]
                break
        
        if ai_opening:
            logger.info(f"[session/start] Reutilizando sesión activa para team {body.team_id}")
            return StartSessionResponse(
                session_id=existing_session.session_id,
                mode=existing_session.mode,
                ai_opening_message=ai_opening,
                messages=existing_session.messages,
                quota=quota,
            )
        else:
            logger.warning(f"[session/start] Sesión {existing_session.session_id} encontrada pero sin historial. Se creará una nueva.")
            session_store.delete(existing_session.session_id)

    session = session_store.create(
        team_id=body.team_id,
        mode=mode,
        analysis_result=actual_analysis,
        proposal_summary=body.proposal_summary,
        team_members=body.team_members,
    )

    quota_client.register_session(
        user_id=body.team_id,
        session_data={
            "session_id": session.session_id,
            "verdict": "approved" if approved else "rejected",
            "proposal_summary": body.proposal_summary[:500],
            "analysis_json": actual_analysis,
        },
    )

    if mode == "defense":
        opening_prompt = (
            "El equipo acaba de ver que su proyecto fue pre-aprobado. "
            "Abre la sesión presentándote estrictamente como la IA 'Corvus Evaluator' (NO inventes nombres humanos para ti) "
            "y lanza tu primera pregunta difícil sobre el punto más débil que identificas en el proyecto. Sé directo y específico. "
            "Recuerda incluir al final '[SCORE: 0/100]'."
        )
    else:
        score = actual_analysis.get("innovation_index", {}).get("score", 0)
        risk = actual_analysis.get("semantic_collision_risk", {}).get("alert_type", "")
        opening_prompt = (
            f"El equipo acaba de recibir el rechazo de su propuesta (score: {score}%, riesgo: {risk}). "
            "Abre la sesión presentándote estrictamente como la IA 'Corvus Advisor' (NO inventes nombres humanos para ti), "
            "explica brevemente las razones principales del rechazo y pregúntales qué parte les gustaría entender mejor primero."
        )

    messages = session.to_ollama_messages()
    messages.append({"role": "user", "content": opening_prompt})

    async def _do_start_chat():
        try:
            ai_opening = await asyncio.to_thread(chat_with_groq, messages, 0.6)
            session.add_message("assistant", ai_opening)
            return ai_opening
        except Exception as e:
            logger.error(f"[session/start] Error generando apertura: {e}")
            session_store.delete(session.session_id)
            raise HTTPException(status_code=500, detail="Error generando el mensaje inicial de la IA.")

    ai_opening = await llm_queue.enqueue(1, _do_start_chat())

    return StartSessionResponse(
        session_id=session.session_id,
        mode=mode,
        ai_opening_message=ai_opening,
        messages=session.messages,
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

    session.add_message("user", body.user_message, body.student_name)

    # Use reminder-injected messages to prevent context drift / name hallucination
    full_messages = session.to_messages_with_reminder()

    async def _do_chat():
        try:
            ai_response = await asyncio.to_thread(chat_with_groq, full_messages, 0.7, body.groq_model)
            
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

        # Intentar con Gemini primero para nombres (gratis, alto rate limit, menos repeticiones)
        try:
            from app.api.gemini_client import generate_text_with_gemini
            from app.config import settings
            logger.info("[generate-name] Intentando con Gemini...")
            gemini_key = getattr(settings, "GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
            if not gemini_key:
                 raise Exception("GEMINI_API_KEY no está configurada")
            raw_response = await generate_text_with_gemini(system_prompt, body.prompt, api_key=gemini_key)
        except Exception as e:
            logger.warning(f"[generate-name] Gemini falló ({e}). Failover a Groq/Ollama...")

        # Si Gemini falló, intentar con el proveedor configurado
        if raw_response is None and body.provider == "groq":
            try:
                from app.api.groq_client import generate_text_with_groq
                logger.info(f"[generate-name] Intentando con Groq usando {body.groq_model}...")
                raw_response = await asyncio.to_thread(
                    generate_text_with_groq, system_prompt, body.prompt, body.groq_model
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
                    system_prompt=system_prompt,
                    json_format=False
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
                logger.info(f"[generate-rag-summary] Intentando con Groq usando {body.groq_model}...")
                raw_response = await asyncio.to_thread(
                    generate_text_with_groq, system_prompt, user_prompt, body.groq_model
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
                    system_prompt=system_prompt,
                    json_format=False
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
                logger.info(f"[analyze-homework] Intentando con Groq usando {body.groq_model}...")
                raw_response = await asyncio.to_thread(
                    analyze_with_groq, system_prompt, user_prompt, body.groq_model
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

@router.post("/generate-career-skills", response_model=GenerateCareerSkillsResponse)
async def generate_career_skills(body: GenerateCareerSkillsRequest):
    """
    Genera 100 habilidades para una carrera universitaria usando Groq o Ollama, con su respectivo peso.
    """
    system_prompt = (
        "Eres un experto en currículas universitarias y orientación vocacional. "
        "Se te dará el nombre de una carrera universitaria y debes devolver un JSON array "
        "con exactamente 100 habilidades (skills) técnicas, blandas y conocimientos clave "
        "que adquiere un egresado de esa carrera. Deben ser cortas (ej. 'Python', 'Liderazgo'). "
        "Para cada habilidad asigna un 'weight' (entero del 1 al 10) según la importancia de la habilidad para esa carrera. "
        "RESPONDE ÚNICAMENTE CON UN JSON ARRAY con este formato exacto: "
        "[{\"name\": \"Python\", \"weight\": 9}, {\"name\": \"Liderazgo\", \"weight\": 7}]"
    )
    user_prompt = body.career_name

    async def _do_generate():
        raw_response = None
        if body.provider == "groq":
            try:
                from app.api.groq_client import generate_text_with_groq
                logger.info(f"[generate-career-skills] Intentando con Groq para {body.career_name} usando {body.groq_model}...")
                raw_response = await asyncio.to_thread(
                    generate_text_with_groq, system_prompt, user_prompt, body.groq_model
                )
            except Exception as e:
                logger.warning(f"[generate-career-skills] Groq falló ({e}). Failover a Ollama...")

        if raw_response is None:
            if not ollama_client.check_health():
                return {"skills": []}
            try:
                raw_response = await ollama_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt
                )
            except Exception as e:
                logger.error(f"[generate-career-skills] Error con Ollama: {e}")
                return {"skills": []}

        try:
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.startswith("```"): cleaned = cleaned[3:]
            if cleaned.endswith("```"): cleaned = cleaned[:-3]
            
            data = json.loads(cleaned)
            
            if isinstance(data, list):
                # Ensure each element is a dict with name and weight
                processed = [{"name": item.get("name", "Unknown"), "weight": item.get("weight", 5)} if isinstance(item, dict) else {"name": str(item), "weight": 5} for item in data]
                return {"skills": processed}
            elif isinstance(data, dict) and "skills" in data:
                processed = [{"name": item.get("name", "Unknown"), "weight": item.get("weight", 5)} if isinstance(item, dict) else {"name": str(item), "weight": 5} for item in data["skills"]]
                return {"skills": processed}
            else:
                return {"skills": []}
        except Exception as e:
            logger.error(f"[generate-career-skills] Parse error: {e}")
            import re
            matches = re.search(r'\[(.*?)\]', raw_response, re.DOTALL)
            if matches:
                try:
                    skills_arr = json.loads(f"[{matches.group(1)}]")
                    processed = [{"name": item.get("name", "Unknown"), "weight": item.get("weight", 5)} if isinstance(item, dict) else {"name": str(item), "weight": 5} for item in skills_arr]
                    return {"skills": processed}
                except:
                    pass
            return {"skills": []}

    return await llm_queue.enqueue(3, _do_generate())

@router.post("/validate-idea-quick")
async def validate_idea_quick(body: ValidateIdeaQuickRequest):
    """
    Evalúa una idea de proyecto integrador verificando reglas del profesor,
    posibles colisiones con otros proyectos y viabilidad general.
    Esta evaluación debe ser súper rápida (idealmente de 1 a 3 párrafos).
    """
    provider = body.provider or "ollama"

    system_prompt = (
        "Eres un Asesor Técnico de Proyectos Integradores en una universidad. "
        "Tu objetivo es dar una retroalimentación extremadamente breve, directa y profesional "
        "(máximo 3 párrafos) sobre la viabilidad de la idea de proyecto que propone el alumno."
    )

    user_prompt = f"El alumno propone la siguiente idea:\n\"{body.idea}\"\n\n"

    if body.blocked_topics or body.blocked_techs:
        user_prompt += "Ten en cuenta que el profesor ha PROHIBIDO estrictamente lo siguiente:\n"
        if body.blocked_topics:
            user_prompt += f"- Temas bloqueados: {', '.join(body.blocked_topics)}\n"
        if body.blocked_techs:
            user_prompt += f"- Tecnologías bloqueadas: {', '.join(body.blocked_techs)}\n"
        user_prompt += "Si la idea usa algo de esto, debes rechazarla inmediatamente.\n\n"

    if body.similar_projects:
        user_prompt += "Existen proyectos anteriores que se parecen mucho a esta idea:\n"
        for p in body.similar_projects:
            title = p.get("title", "Proyecto")
            desc = p.get("description", "")
            user_prompt += f"- {title}: {desc}\n"
        user_prompt += "Advierte al alumno sobre el riesgo de similitud y pídele que innove más si se parece demasiado.\n\n"

    user_prompt += (
        "Da tu veredicto final indicando si la idea parece viable, "
        "si requiere ajustes o si debe ser descartada por violar las reglas."
    )

    async def _do_generate():
        if provider.lower() in ["groq", "grok"]:
            from app.api.groq_client import generate_text_with_groq
            try:
                # Groq client is synchronous, so we use asyncio.to_thread if we want, or just call it directly.
                # Actually generate_text_with_groq is synchronous, let's wrap it in to_thread.
                response = await asyncio.to_thread(generate_text_with_groq, system_prompt, user_prompt, body.groq_model)
                return {"result": response}
            except Exception as e:
                logger.error(f"[validate-idea-quick] Error con Groq: {e}")
                return {"result": "Error al evaluar la idea con Groq."}

        if not ollama_client.check_health():
            return {"result": "El motor de IA local no está disponible."}
        try:
            response = await ollama_client.generate(prompt=user_prompt, system_prompt=system_prompt, json_format=False)
            return {"result": response}
        except Exception as e:
            logger.error(f"[validate-idea-quick] Error con Ollama: {e}")
            return {"result": "Error al evaluar la idea con Ollama."}

    return await llm_queue.enqueue(1, _do_generate())

@router.get("/session/{session_id}/messages")
async def get_session_messages(session_id: str):
    existing_session = session_store.get(session_id)
    if not existing_session:
        raise HTTPException(status_code=404, detail="Sesión no encontrada.")
    return {"messages": existing_session.messages}

@router.get("/debug-last-prompt")
async def debug_last_prompt():
    """
    Endpoint de diagnóstico: devuelve el último prompt enviado a Groq/Ollama
    con tamaños y previews. No ejecuta nada, solo inspecciona.
    """
    return {
        "status": "ok",
        "last_prompt": _last_prompt_debug if _last_prompt_debug else {"message": "No se ha enviado ningún prompt todavía."}
    }
