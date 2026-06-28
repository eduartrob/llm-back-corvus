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

# ─── Prompt de análisis de propuesta (reutilizado del clustering service) ───

def build_analysis_prompt(proposal_text: str, context_text: str, project_name: str, top_project_name: str, max_sim_pct: float, risk_level: str) -> str:
    """
    Genera el prompt completo con variables dinámicas.
    COPIA EXACTA del ollama_service.py local que ya fue probado y da resultados correctos.
    """
    return f"""Eres un evaluador académico de proyectos universitarios dentro del sistema AcadeRAG.

=== TU ÚNICA TAREA ===
Redactar un DICTAMEN COMPLETO sobre el proyecto "{project_name}".
TODAS tus métricas (innovation_index, quality_metrics, recommendations, verdict) deben referirse EXCLUSIVAMENTE a "{project_name}".
PROHIBIDO dar recomendaciones sobre los proyectos del historial. Esos proyectos son SOLO para detectar plagio.

=== PROYECTO A EVALUAR: "{project_name}" ===
{proposal_text}
=== FIN DE "{project_name}" ===

=== HISTORIAL (SOLO LECTURA PARA DETECTAR PLAGIO, NO EVALUAR) ===
{context_text}
=== FIN DEL HISTORIAL ===

=== REGLAS DE COLISIÓN ===
El sistema (Python) ya calculó matemáticamente que el riesgo de colisión es: {risk_level.upper()} (Similitud máxima: {max_sim_pct}%).
Tu trabajo es escribir la "explanation" justificando POR QUÉ el sistema dio este riesgo.
DEBES mencionar explícitamente el proyecto '{top_project_name}'. Además, DEBES extraer y escribir LITERALMENTE al menos 2 conceptos, tecnologías o palabras clave exactas que ambos proyectos tienen en común para justificar ese {max_sim_pct}%.
- Si el riesgo es ALTO, enumera qué partes son idénticas al historial y por qué parece plagio.
- Si el riesgo es MEDIO o BAJO, escribe exactamente qué conceptos comparten con '{top_project_name}' pero destaca por qué "{project_name}" es diferente y original.

RECUERDA ANTES DE ESCRIBIR EL JSON: ¿Tus recomendaciones están dirigidas al autor de "{project_name}"? Si no, corrígelas.

INSTRUCCIONES FINALES DE ESTRUCTURA JSON:
Tu salida debe ser ÚNICA y EXCLUSIVAMENTE un documento JSON válido, sin ningún texto Markdown ni comentarios fuera de él. El JSON llenará un Dashboard UI.
Respeta EXACTAMENTE esta estructura y sigue las reglas para cada campo:
{{
  "innovation_index": {{
    "score": <número 0-100>,
    "label": "<Usa exactamente una de estas: Excepcional | Muy Bueno | Aceptable | Tradicional>"
  }},
  "quality_metrics": {{
    "academic_rigor": <número 0-100 evaluando citas y referencias>,
    "technical_relevance": <número 0-100 evaluando la modernidad de la tecnología propuesta>,
    "structural_clarity": <número 0-100 evaluando la redacción y organización>
  }},
  "semantic_collision_risk": {{
    "alert_type": "<Alerta Roja | Alerta Amarilla | Falsa Alarma>",
    "explanation": "<Redacta aquí tu explicación detallada. OBLIGATORIO: Menciona explícitamente a '{top_project_name}' y escribe 2 tecnologías o conceptos exactos que comparten.>"
  }},
  "recommendations": [
    {{
      "icon": "<elige uno: code, lock, fact_check, architecture, library_books>",
      "title": "<Redacta un título corto y útil para tu primera recomendación>",
      "description": "<Redacta aquí la instrucción detallada para el alumno>"
    }},
    {{
      "icon": "<elige uno: code, lock, fact_check, architecture, library_books>",
      "title": "<Redacta un título corto y útil para tu segunda recomendación>",
      "description": "<Redacta aquí la instrucción detallada para el alumno>"
    }},
    {{
      "icon": "<elige uno: code, lock, fact_check, architecture, library_books>",
      "title": "<Redacta un título corto y útil para tu tercera recomendación>",
      "description": "<Redacta aquí la instrucción detallada para el alumno>"
    }}
  ],
  // REGLA ESTRICTA: DEBES GENERAR UN MÍNIMO DE 3 RECOMENDACIONES. NUNCA MENOS DE 3.
  "verdict": "<Breve resumen del dictamen>",
  "approved": <booleano true o false>
}}
"""


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
# ENDPOINT 1.5: Analyze Blue Ocean Niche
# ────────────────────────────────────────────────────────────────────────────
@router.post("/analyze-blue-ocean")
async def analyze_blue_ocean(body: BlueOceanRequest):
    """
    Recibe la información de un Océano Azul inexplorado y genera un 
    análisis estructurado con hallazgo, sugerencias y métricas.
    """
    if not ollama_client.check_health():
        raise HTTPException(status_code=503, detail="El motor de IA (Ollama) no está disponible.")

    prompt = f"""Eres un asesor académico de innovación tecnológica. Se te presenta un nicho de investigación detectado como "Océano Azul" (baja colisión semántica con proyectos históricos).

Nicho: {body.title}
Descripción: {body.description}
Categoría: {body.category}

Responde ÚNICAMENTE con un JSON válido con esta estructura exacta y sin formato markdown:
{{
  "hallazgo_principal": "string - 2 o 3 oraciones de por qué nadie lo ha tocado y qué vacío existe",
  "sugerencias": [
    {{"titulo": "Enfoque Mixto", "descripcion": "breve...", "tipo": "Recomendado"}},
    {{"titulo": "Estudio de Caso", "descripcion": "breve...", "tipo": "Alternativo"}}
  ],
  "metricas": {{
    "originalidad": 90,
    "disponibilidad_datos": 70,
    "relevancia_academica": 85
  }}
}}
"""

    try:
        raw_response = await ollama_client.generate(prompt=prompt)
        
        # Limpiar markdown de la respuesta de ollama
        cleaned_response = raw_response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        import json
        return json.loads(cleaned_response)
        
    except Exception as e:
        logger.error(f"[analyze-blue-ocean] Error: {e}")
        # Fallback en caso de que el LLM falle o el JSON sea inválido
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

    # Construir contexto de todos los proyectos similares (como pidió el usuario para mejorar el razonamiento)
    context_text = ""
    for i, proj in enumerate(body.similar_projects):
        sim_pct = proj.get("similarity_pct", 0)
        context_text += (
            f"\n--- Proyecto Existente {i+1} ---\n"
            f"Título: {proj.get('title', 'Desconocido')}\n"
            f"Similitud Matemática (ChromaDB): {sim_pct:.1f}%\n"
            f"Contenido: {proj.get('content', '')}\n"
        )

    # Generar el Prompt Magistral con variables dinámicas
    prompt = build_analysis_prompt(
        proposal_text=body.proposal_text,
        context_text=context_text,
        project_name=body.project_name,
        top_project_name=body.top_project_name,
        max_sim_pct=body.max_sim_pct,
        risk_level=body.risk_level,
    )

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
