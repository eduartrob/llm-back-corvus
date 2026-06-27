import logging
import requests
from app.config import settings

logger = logging.getLogger(__name__)


class QuotaClient:
    """
    Consulta el microservicio de autenticación para verificar
    cuántas sesiones LLM ha usado un usuario.
    """

    def __init__(self):
        self.auth_url = settings.AUTH_SERVICE_URL

    def get_quota(self, user_id: str) -> dict:
        """
        Retorna: { sessions_used: int, limit: int, unlimited: bool, can_create: bool }
        Si UNLIMITED_SESSIONS=True, retorna siempre can_create=True sin consultar la DB.
        """
        if settings.UNLIMITED_SESSIONS:
            return {
                "sessions_used": 0,
                "limit": -1,
                "unlimited": True,
                "can_create": True,
            }

        try:
            response = requests.get(
                f"{self.auth_url}/internal/users/{user_id}/llm-quota",
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                sessions_used = data.get("sessions_used", 0)
                limit = settings.FREE_SESSION_LIMIT
                return {
                    "sessions_used": sessions_used,
                    "limit": limit,
                    "unlimited": False,
                    "can_create": sessions_used < limit,
                }
            else:
                logger.warning(f"[QuotaClient] Auth service respondió {response.status_code}. Bloqueando por seguridad.")
                return {"sessions_used": 0, "limit": settings.FREE_SESSION_LIMIT, "unlimited": False, "can_create": False}
        except Exception as e:
            logger.error(f"[QuotaClient] Error consultando cuota: {e}. Bloqueando por seguridad.")
            return {"sessions_used": 0, "limit": settings.FREE_SESSION_LIMIT, "unlimited": False, "can_create": False}

    def register_session(self, user_id: str, session_data: dict) -> bool:
        """
        Registra una nueva sesión LLM en la DB del auth service.
        Si UNLIMITED_SESSIONS=True, no registra (modo testing sin datos de cuota).
        """
        if settings.UNLIMITED_SESSIONS:
            return True
        try:
            response = requests.post(
                f"{self.auth_url}/internal/users/{user_id}/llm-sessions",
                json=session_data,
                timeout=10,
            )
            return response.status_code in (200, 201)
        except Exception as e:
            logger.error(f"[QuotaClient] Error registrando sesión: {e}")
            return False


quota_client = QuotaClient()
