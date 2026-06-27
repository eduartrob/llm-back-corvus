import json
import logging
from fastapi import APIRouter, HTTPException, Header
from typing import Optional

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

# ─── Prompt de análisis de propuesta (reutilizado del clustering service) ───

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


# ────────────────────────────────────────────────────────────────────────────
# ENDPOINT 1: Health Check
# ────────────────────────────────────────────────────────────────────────────
@router.get("/health")
async def health():
    ollama_ok = ollama_client.check_health()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "service": "llm-back-corvus",
        "ollama": "connected" if ollama_ok else "unreachable",
        "model": ollama_client.model,
    }


# ────────────────────────────────────────────────────────────────────────────
# ENDPOINT 2: Analyze Proposal (llamado por integratorClustering)
# ────────────────────────────────────────────────────────────────────────────
@router.post("/analyze-proposal")
async def analyze_proposal(body: AnalyzeProposalRequest):
    """
    Recibe el texto de la propuesta y los proyectos similares del clustering.
    Devuelve el dictamen completo en JSON estructurado.
    Llamado internamente por integratorProjectClustering-back-corvus.
    """
    if not ollama_client.check_health():
        raise HTTPException(status_code=503, detail="El motor de IA (Ollama) no está disponible.")

    # Construir contexto de proyectos similares
    context_text = ""
    for i, proj in enumerate(body.similar_projects):
        sim_pct = proj.get("similarity_pct", 0)
        context_text += (
            f"\n--- Proyecto Existente {i+1} ---\n"
            f"Título: {proj.get('title', 'Desconocido')}\n"
            f"Similitud: {sim_pct:.1f}%\n"
            f"Contenido: {proj.get('content', '')}\n"
        )

    prompt = (
        f"--- NUEVA PROPUESTA ---\n{body.proposal_text}\n--- FIN PROPUESTA ---\n\n"
        f"--- HISTORIAL SIMILARES (SOLO REFERENCIA) ---\n{context_text}\n--- FIN HISTORIAL ---"
    )

    try:
        raw_response = await ollama_client.generate(
            prompt=prompt,
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
        )
        result = json.loads(raw_response)
        return result
    except json.JSONDecodeError:
        logger.error(f"[analyze-proposal] Ollama no devolvió JSON válido: {raw_response}")
        raise HTTPException(status_code=500, detail="El modelo no devolvió un formato válido.")
    except Exception as e:
        logger.error(f"[analyze-proposal] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ────────────────────────────────────────────────────────────────────────────
# ENDPOINT 3: Iniciar sesión conversacional
# ────────────────────────────────────────────────────────────────────────────
@router.post("/session/start", response_model=StartSessionResponse)
async def start_session(
    body: StartSessionRequest,
    x_user_data: Optional[str] = Header(None),
):
    """
    Crea una sesión conversacional según el resultado del análisis:
    - approved=true  → Modo DEFENSA (la IA lleva la contraria)
    - approved=false → Modo RECHAZO (la IA explica y asesora)
    Verifica la cuota del usuario antes de crear la sesión.
    """
    # 1. Verificar cuota
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

    # 2. Determinar modo
    approved = body.analysis_result.get("approved", False)
    mode = "defense" if approved else "rejection"

    # 3. Crear sesión en RAM
    session = session_store.create(
        user_id=body.user_id,
        mode=mode,
        analysis_result=body.analysis_result,
        proposal_summary=body.proposal_summary,
    )

    # 4. Registrar en DB (para control de cuotas en producción)
    quota_client.register_session(
        user_id=body.user_id,
        session_data={
            "session_id": session.session_id,
            "verdict": "approved" if approved else "rejected",
            "proposal_summary": body.proposal_summary[:500],
            "analysis_json": body.analysis_result,
        },
    )

    # 5. Generar el mensaje de apertura de la IA
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
        # Guardar el intercambio en el historial (sin el prompt interno)
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


# ────────────────────────────────────────────────────────────────────────────
# ENDPOINT 4: Enviar mensaje a sesión activa
# ────────────────────────────────────────────────────────────────────────────
@router.post("/session/message", response_model=SessionMessageResponse)
async def session_message(body: SessionMessageRequest):
    """
    Envía un mensaje del alumno a la sesión activa.
    La IA responde manteniendo todo el historial de la conversación.
    """
    session = session_store.get(body.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Sesión no encontrada o expirada (las sesiones expiran tras 30 minutos de inactividad).",
        )

    if not ollama_client.check_health():
        raise HTTPException(status_code=503, detail="El motor de IA no está disponible en este momento.")

    # Añadir el mensaje del alumno al historial
    session.add_message("user", body.user_message)

    # Construir el historial completo para Ollama
    full_messages = session.to_ollama_messages()

    try:
        ai_response = await ollama_client.chat(full_messages, temperature=0.6)
        session.add_message("assistant", ai_response)
    except Exception as e:
        logger.error(f"[session/message] Error: {e}")
        # Revertir el mensaje del usuario para no corromper el historial
        session.messages.pop()
        raise HTTPException(status_code=500, detail="Error procesando tu mensaje. Intenta de nuevo.")

    return SessionMessageResponse(
        ai_message=ai_response,
        session_id=body.session_id,
    )
