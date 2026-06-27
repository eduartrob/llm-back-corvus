import asyncio
import logging
import requests
from app.config import settings

logger = logging.getLogger(__name__)

# Semáforo para limitar concurrencia a Ollama (evita saturar la CPU del VPS)
_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

# Endpoints de ADMINISTRACIÓN que nunca se deben exponer hacia afuera
_BLOCKED_ENDPOINTS = {"/api/delete", "/api/pull", "/api/push", "/api/copy", "/api/create", "/api/blobs"}


class OllamaClient:
    """
    Proxy seguro hacia Ollama.
    - Solo permite el modelo configurado en ALLOWED_MODEL.
    - Solo usa /api/generate y /api/chat (no endpoints de gestión).
    - Limita las peticiones simultáneas con un semáforo.
    """

    def __init__(self):
        self.host = settings.OLLAMA_HOST
        self.model = settings.ALLOWED_MODEL

    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _validate_model(self, model: str):
        if model != self.model:
            raise ValueError(f"Modelo '{model}' no permitido. Solo se permite '{self.model}'.")

    async def generate(self, prompt: str, system_prompt: str = "") -> str:
        """
        Genera una respuesta en modo JSON estructurado.
        Usado por analyze-proposal (respuesta JSON completa).
        """
        async with _semaphore:
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": full_prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0.2},
                    },
                    timeout=900,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "{}")
            except Exception as e:
                logger.error(f"[OllamaClient] Error en generate: {e}")
                raise

    async def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        """
        Continúa una conversación multi-turno.
        Usado por session/message (modo Defensa o modo Rechazo).
        messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        async with _semaphore:
            try:
                response = await asyncio.to_thread(
                    requests.post,
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {"temperature": temperature},
                    },
                    timeout=900,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")
            except Exception as e:
                logger.error(f"[OllamaClient] Error en chat: {e}")
                raise


ollama_client = OllamaClient()
