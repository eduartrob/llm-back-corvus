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

ANALYSIS_SYSTEM_PROMPT = """Eres un estricto comité evaluador de proyectos universitarios en AcadeRAG.
Evalúa EXCLUSIVAMENTE la nueva propuesta. El historial es solo referencia para detectar plagio.

REGLAS DE COLISIÓN:
- Similitud > 50% con idea idéntica → Alerta Roja
- Similitud 20-50% con diferenciadores → Falsa Alarma
- Similitud 20-50% sin diferenciadores → Alerta Amarilla  
- Similitud > 90% → Plagio (innovation_index: 0, approved: false)

Responde ÚNICAMENTE con un JSON válido sin markdown ni comentarios con esta estructura exacta:
{
  "innovation_index": { "score": <0-100>, "label": "<Excepcional|Muy Bueno|Aceptable|Tradicional>" },
  "quality_metrics": { "academic_rigor": <0-100>, "technical_relevance": <0-100>, "structural_clarity": <0-100> },
  "semantic_collision_risk": { "alert_type": "<Alerta Roja|Alerta Amarilla|Falsa Alarma>", "explanation": "<análisis detallado>" },
  "recommendations": [{ "icon": "<code|lock|fact_check|architecture|library_books>", "title": "<título>", "description": "<descripción>" }],
  "verdict": "<resumen del dictamen>",
  "approved": <true|false>
}"""

def build_groq_analysis_prompt(proposal_text: str, context_text: str, project_name: str, top_project_name: str, max_sim_pct: float, risk_level: str) -> tuple[str, str]:
    system_prompt = f"""Eres un estricto evaluador académico de proyectos universitarios dentro del sistema AcadeRAG.

=== TU ÚNICA TAREA ===
Redactar un DICTAMEN COMPLETO sobre el proyecto.
TODAS tus métricas deben referirse EXCLUSIVAMENTE al proyecto evaluado.
PROHIBIDO dar recomendaciones sobre los proyectos del historial. Esos proyectos son SOLO para detectar plagio.

=== REGLAS ESTRICTAS DE EVALUACIÓN ===
1. COLISIÓN: El sistema ya calculó matemáticamente que el riesgo de colisión es: {risk_level.upper()} (Similitud: {max_sim_pct}%). En el campo 'explanation' de 'semantic_collision_risk', DEBES justificar detalladamente por qué el enfoque es distinto (o similar) al proyecto '{top_project_name}'.
2. SECCIONES FALTANTES: Si al documento le faltan secciones clave (ej. no tiene objetivos claros, no tiene problemática, no tiene variables), DEBES castigar severamente las métricas de 'academic_rigor' y 'structural_clarity'.
3. RECOMENDACIONES: Genera exactamente 4 recomendaciones técnicas. Si faltan secciones, la primera recomendación DEBE ser pedir que agreguen lo que falta.

INSTRUCCIONES FINALES DE ESTRUCTURA JSON:
Tu salida debe ser ÚNICA y EXCLUSIVAMENTE un documento JSON válido. No devuelvas ningún texto de relleno ni uses "textos de ejemplo", DEBES LLENAR el JSON con tu propio análisis real y profundo.

El JSON debe tener EXACTAMENTE estas claves y tipos de datos:
- "innovation_index": objeto con "score" (número del 0 al 100) y "label" (string).
- "quality_metrics": objeto con "academic_rigor" (número), "technical_relevance" (número) y "structural_clarity" (número).
- "semantic_collision_risk": objeto con "alert_type" (string) y "explanation" (string).
- "recommendations": arreglo de 4 objetos, donde cada uno tiene "icon" (string: elige entre 'code', 'lock', 'fact_check', 'architecture' o 'library_books'), "title" (string) y "description" (string largo).
- "verdict": string (un breve resumen).
- "approved": booleano (true o false).
"""

    user_prompt = f"""=== PROYECTO A EVALUAR: "{project_name}" ===
{proposal_text}
=== FIN DE "{project_name}" ===

=== HISTORIAL (SOLO LECTURA PARA DETECTAR PLAGIO, NO EVALUAR) ===
{context_text}
=== FIN DEL HISTORIAL ===
"""
    
    return system_prompt, user_prompt

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
        system_prompt = "Eres un experto analista de datos e investigador académico. Genera un análisis JSON detallado para un tema de Océano Azul."
        user_prompt = f"""Analiza este nicho de océano azul (baja colisión semántica).
Título: {body.title}
Descripción: {body.description}
Categoría: {body.category}

Genera un JSON con tres sugerencias de innovación, un hallazgo principal, y métricas.
Estructura JSON estricta (no uses backticks):
{{
    "hallazgo_principal": "string",
    "sugerencias": [
        {{"titulo": "string", "descripcion": "string", "tipo": "string"}}
    ],
    "metricas": {{
        "originalidad": 85,
        "disponibilidad_datos": 60,
        "relevancia_academica": 90
    }}
}}"""
        
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
            return {
                "hallazgo_principal": "Este tema presenta una oportunidad única por su baja colisión con los registros académicos actuales.",
                "sugerencias": [
                    {"titulo": "Investigación Cuantitativa", "descripcion": "Desarrollar métricas base.", "tipo": "Recomendado"},
                    {"titulo": "Estudio Exploratorio", "descripcion": "Evaluar viabilidad en campo.", "tipo": "Alternativo"}
                ],
                "metricas": {
                    "originalidad": 85,
                    "disponibilidad_datos": 60,
                    "relevancia_academica": 90
                }
            }

    # Prioridad baja (10) para tareas de fondo
    return await llm_queue.enqueue(10, _do_analysis())

def build_ollama_analysis_prompt(proposal_text: str, context_text: str, project_name: str, top_project_name: str, max_sim_pct: float, risk_level: str) -> tuple[str, str]:
    system_prompt = f"""Eres un estricto evaluador académico de proyectos universitarios dentro del sistema AcadeRAG.
TU ÚNICA TAREA es redactar un DICTAMEN COMPLETO sobre la nueva propuesta.

REGLAS ESTRICTAS DE EVALUACIÓN:
1. COLISIÓN: El sistema ya calculó matemáticamente que el riesgo de colisión es: {risk_level.upper()} (Similitud: {max_sim_pct}%). Justifica detalladamente en 'explanation' por qué el enfoque es distinto (o similar) al proyecto '{top_project_name}'.
2. RECOMENDACIONES: Genera exactamente 4 recomendaciones técnicas sobre cómo mejorar.
3. CALIFICACIONES: Las métricas deben ser números reales (0 a 100) basados en tu evaluación real. NO uses los textos de ejemplo de la plantilla.

INSTRUCCIONES FINALES DE ESTRUCTURA JSON:
Tu salida debe ser ÚNICA y EXCLUSIVAMENTE un documento JSON válido. Responde ÚNICAMENTE con esta estructura exacta, reemplazando los valores en corchetes angulares por tus valores reales:
{{
  "innovation_index": {{ "score": <0-100>, "label": "<Excepcional|Muy Bueno|Aceptable|Tradicional>" }},
  "quality_metrics": {{ "academic_rigor": <0-100>, "technical_relevance": <0-100>, "structural_clarity": <0-100> }},
  "semantic_collision_risk": {{ "alert_type": "<Alerta Roja|Alerta Amarilla|Falsa Alarma>", "explanation": "<análisis detallado>" }},
  "recommendations": [{{ "icon": "<code|lock|fact_check|architecture|library_books>", "title": "<título>", "description": "<descripción>" }}],
  "verdict": "<resumen del dictamen>",
  "approved": <true|false>
}}"""

    user_prompt = f"""=== PROYECTO A EVALUAR: "{project_name}" ===
{proposal_text}
=== FIN DE "{project_name}" ===

=== HISTORIAL (SOLO LECTURA PARA DETECTAR PLAGIO, NO EVALUAR) ===
{context_text}
=== FIN DEL HISTORIAL ===
"""
    return system_prompt, user_prompt

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
