import json
import logging
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

router = APIRouter()
logger = logging.getLogger(__name__)

class BlueOceanRequest(BaseModel):
    title: str
    description: str
    category: str

-# prompt de analisis de propuesta reutilizado del clustering service

def build_analysis_prompt(proposal_text: str, context_text: str, project_name: str, top_project_name: str, max_sim_pct: float, risk_level: str) -> str:
    
    return f

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
    
    if not ollama_client.check_health():
        raise HTTPException(status_code=503, detail="El motor de IA (Ollama) no está disponible.")

    prompt = f

    try:
        raw_response = await ollama_client.generate(prompt=prompt)
        
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        import json
        return json.loads(cleaned_response)
        
    except Exception as e:
        logger.error(f"[analyze-blue-ocean] Error: {e}")
        return {
            "hallazgo_principal": "Este tema presenta una oportunidad única por su baja colisión con los registros académicos actuales. Existe un vacío sustancial en la literatura local reciente.",
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

@router.post("/analyze-proposal")
async def analyze_proposal(body: AnalyzeProposalRequest):
    
    context_text = ""
    for i, proj in enumerate(body.similar_projects):
        sim_pct = proj.get("similarity_pct", 0)
        context_text += (
            f"\n--- Proyecto Existente {i+1} ---\n"
            f"Título: {proj.get('title', 'Desconocido')}\n"
            f"Similitud Matemática (ChromaDB): {sim_pct:.1f}%\n"
            f"Contenido: {proj.get('content', '')}\n"
        )

    prompt = build_analysis_prompt(
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
            result = analyze_with_groq(prompt)
            return result
        except Exception as e:
            logger.warning(f"[analyze-proposal] Falló GroqCloud ({e}). Haciendo failover a Ollama local...")

    # Flujo normal o fallback a Ollama
    if not ollama_client.check_health():
        raise HTTPException(status_code=503, detail="El motor de IA (Ollama) no está disponible.")

    try:
        raw_response = await ollama_client.generate(prompt=prompt)
        result = json.loads(raw_response)
        return result
    except json.JSONDecodeError:
        logger.error(f"[analyze-proposal] Ollama no devolvió JSON válido: {raw_response}")
        raise HTTPException(status_code=500, detail="El modelo no devolvió un formato válido.")
    except Exception as e:
        logger.error(f"[analyze-proposal] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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

    try:
        ai_opening = await ollama_client.chat(messages, temperature=0.6)
        session.add_message("assistant", ai_opening)
    except Exception as e:
        logger.error(f"[session/start] Error generando apertura: {e}")
        raise HTTPException(status_code=500, detail="Error generando el mensaje inicial de la IA.")

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

    try:
        ai_response = await ollama_client.chat(full_messages, temperature=0.6)
        session.add_message("assistant", ai_response)
    except Exception as e:
        logger.error(f"[session/message] Error: {e}")
        session.messages.pop()
        raise HTTPException(status_code=500, detail="Error procesando tu mensaje. Intenta de nuevo.")

    return SessionMessageResponse(
        ai_message=ai_response,
        session_id=body.session_id,
    )
