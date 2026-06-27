import uuid
import time
import threading
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# TTL en segundos: las sesiones inactivas se limpian tras 30 minutos
SESSION_TTL_SECONDS = 30 * 60

DEFENSE_SYSTEM_PROMPT = """Eres un riguroso comité evaluador universitario llamado "Corvus Evaluator".
Tu rol es hacer de abogado del diablo para un proyecto que YA FUE APROBADO.
Debes cuestionar de forma inteligente y constructiva los puntos débiles del proyecto.
Hazte preguntas como: ¿Puede venderse al mercado? ¿Tiene viabilidad técnica real? ¿Cómo se diferencia de la competencia?
Cuando el alumno responda, analiza su respuesta y sigue cuestionando los puntos no resueltos.
Si el alumno da buenas respuestas, reconócelo brevemente y sigue con el siguiente punto débil.
Responde siempre en español. Sé conciso (máximo 3 párrafos por respuesta).
NO repitas preguntas que ya fueron respondidas satisfactoriamente."""

REJECTION_SYSTEM_PROMPT = """Eres un asesor académico constructivo llamado "Corvus Advisor".
El proyecto del alumno fue RECHAZADO. Tu rol es:
1. Explicar con claridad y empatía por qué fue rechazado (usando los datos del análisis).
2. Responder las preguntas del alumno sobre cómo mejorar su propuesta.
3. Sugerir cambios concretos y accionables.
Responde siempre en español. Sé empático pero directo. Máximo 3 párrafos por respuesta."""


class LlmSession:
    def __init__(
        self,
        session_id: str,
        user_id: str,
        mode: str,  # 'defense' | 'rejection'
        analysis_result: dict,
        proposal_summary: str,
    ):
        self.session_id = session_id
        self.user_id = user_id
        self.mode = mode
        self.analysis_result = analysis_result
        self.proposal_summary = proposal_summary
        self.messages: list[dict] = []  # Historial Ollama [{role, content}]
        self.created_at = datetime.utcnow().isoformat()
        self.last_activity = time.time()

    def get_system_prompt(self) -> str:
        analysis_context = (
            f"\nDatos del análisis del proyecto:\n"
            f"- Índice de Innovación: {self.analysis_result.get('innovation_index', {}).get('score', 'N/A')}%\n"
            f"- Rigor Académico: {self.analysis_result.get('quality_metrics', {}).get('academic_rigor', 'N/A')}%\n"
            f"- Relevancia Técnica: {self.analysis_result.get('quality_metrics', {}).get('technical_relevance', 'N/A')}%\n"
            f"- Riesgo de Colisión: {self.analysis_result.get('semantic_collision_risk', {}).get('alert_type', 'N/A')}\n"
            f"- Dictamen: {self.analysis_result.get('verdict', 'N/A')}\n"
            f"\nResumen del proyecto:\n{self.proposal_summary}"
        )
        base = DEFENSE_SYSTEM_PROMPT if self.mode == "defense" else REJECTION_SYSTEM_PROMPT
        return base + analysis_context

    def to_ollama_messages(self) -> list[dict]:
        return [{"role": "system", "content": self.get_system_prompt()}] + self.messages

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.last_activity = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_activity) > SESSION_TTL_SECONDS


class SessionStore:
    """
    Almacén de sesiones en RAM con limpieza automática por TTL.
    """

    def __init__(self):
        self._sessions: dict[str, LlmSession] = {}
        self._lock = threading.Lock()
        self._start_cleanup_thread()

    def create(
        self,
        user_id: str,
        mode: str,
        analysis_result: dict,
        proposal_summary: str,
    ) -> LlmSession:
        session_id = str(uuid.uuid4())
        session = LlmSession(
            session_id=session_id,
            user_id=user_id,
            mode=mode,
            analysis_result=analysis_result,
            proposal_summary=proposal_summary,
        )
        with self._lock:
            self._sessions[session_id] = session
        logger.info(f"[SessionStore] Nueva sesión {session_id} para user {user_id} (modo: {mode})")
        return session

    def get(self, session_id: str) -> Optional[LlmSession]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.is_expired():
                del self._sessions[session_id]
                logger.info(f"[SessionStore] Sesión {session_id} expirada y eliminada.")
                return None
            return session

    def _start_cleanup_thread(self):
        def cleanup():
            while True:
                time.sleep(300)  # Limpiar cada 5 minutos
                expired = []
                with self._lock:
                    for sid, s in self._sessions.items():
                        if s.is_expired():
                            expired.append(sid)
                    for sid in expired:
                        del self._sessions[sid]
                if expired:
                    logger.info(f"[SessionStore] Limpiadas {len(expired)} sesiones expiradas.")

        t = threading.Thread(target=cleanup, daemon=True)
        t.start()


session_store = SessionStore()
