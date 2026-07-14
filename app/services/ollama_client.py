import asyncio
import logging
import requests
from app.config import settings

logger = logging.getLogger(__name__)

_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)

_BLOCKED_ENDPOINTS = {"/api/delete", "/api/pull", "/api/push", "/api/copy", "/api/create", "/api/blobs"}

class OllamaClient:
    

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

    async def generate(self, prompt: str, system_prompt: str = "", json_format: bool = True) -> str:
        
        async with _semaphore:
            try:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2},
                }
                if json_format:
                    payload["format"] = "json"
                if system_prompt:
                    payload["system"] = system_prompt

                response = await asyncio.to_thread(
                    requests.post,
                    f"{self.host}/api/generate",
                    json=payload,
                    timeout=900,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "{}")
            except Exception as e:
                logger.error(f"[OllamaClient] Error en generate: {e}")
                raise

    async def chat(self, messages: list[dict], temperature: float = 0.7) -> str:
        
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
