import uuid
import time
import threading
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 30 * 60

from app.core.prompts import DEFENSE_SYSTEM_PROMPT, REJECTION_SYSTEM_PROMPT

class LlmSession:
    def __init__(
        self,
        session_id: str,
        team_id: str,
        mode: str,  # 'defense' | 'rejection'
        analysis_result: dict,
        proposal_summary: str,
        team_members: Optional[list[str]] = None,
    ):
        self.session_id = session_id
        self.team_id = team_id
        self.mode = mode
        self.analysis_result = analysis_result
        self.proposal_summary = proposal_summary
        self.team_members = team_members
        self.messages: list[dict] = []
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
            f"\nResumen del proyecto:\n{self.proposal_summary}\n"
        )
        if self.team_members:
            analysis_context += f"\nMiembros del equipo: {', '.join(self.team_members)}\n"

        base = DEFENSE_SYSTEM_PROMPT if self.mode == "defense" else REJECTION_SYSTEM_PROMPT
        return base + analysis_context

    def to_ollama_messages(self) -> list[dict]:
        return [{"role": "system", "content": self.get_system_prompt()}] + self.messages

    def add_message(self, role: str, content: str, student_name: Optional[str] = None):
        if student_name and role == "user":
            content = f"[{student_name}]: {content}"
        self.messages.append({"role": role, "content": content})
        self.last_activity = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.last_activity) > SESSION_TTL_SECONDS

class SessionStore:
    

    def __init__(self):
        self._sessions: dict[str, LlmSession] = {}
        self._lock = threading.Lock()
        self._start_cleanup_thread()

    def create(
        self, 
        team_id: str, 
        mode: str, 
        analysis_result: dict, 
        proposal_summary: str, 
        team_members: Optional[list[str]] = None
    ) -> LlmSession:
        
        session_id = str(uuid.uuid4())
        session = LlmSession(
            session_id=session_id,
            team_id=team_id,
            mode=mode,
            analysis_result=analysis_result,
            proposal_summary=proposal_summary,
            team_members=team_members,
        )
        with self._lock:
            self._sessions[session_id] = session
        logger.info(f"[SessionStore] Nueva sesión {session_id} para team {team_id} (modo: {mode})")
        return session

    def get(self, session_id: str) -> Optional[LlmSession]:
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.is_expired():
                del self._sessions[session_id]
                logger.info(f"[SessionStore] Sesión {session_id} expirada y eliminada.")
                return None
            return session

    def get_by_team_id(self, team_id: str) -> Optional[LlmSession]:
        with self._lock:
            for sid, session in list(self._sessions.items()):
                if session.team_id == team_id:
                    if session.is_expired():
                        del self._sessions[sid]
                        logger.info(f"[SessionStore] Sesión {sid} expirada y eliminada.")
                    else:
                        return session
            return None

    def _start_cleanup_thread(self):
        def cleanup():
            while True:
                time.sleep(300)
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
