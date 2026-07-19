import os
import json
import logging
from groq import Groq, RateLimitError

logger = logging.getLogger(__name__)

from app.config import settings

# Configuración del cliente Groq
client = Groq(api_key=settings.GROQ_API_KEY)

FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]

import threading
_rr_lock = threading.Lock()
_rr_index = 0

def get_models_to_try(requested_model: str) -> list[str]:
    global _rr_index
    
    # Si el modelo solicitado NO está en la lista pesada (ej: llama-3.1-8b-instant para chat),
    # lo intentamos primero, y si falla usamos la lista pesada como respaldo normal.
    if requested_model not in FALLBACK_MODELS:
        return [requested_model] + [m for m in FALLBACK_MODELS]
        
    # Si es uno de los modelos pesados, hacemos Round-Robin (1, 2, 3, 1, 2, 3...)
    with _rr_lock:
        start_idx = _rr_index
        _rr_index = (_rr_index + 1) % len(FALLBACK_MODELS)
        
    rotation = []
    for i in range(len(FALLBACK_MODELS)):
        idx = (start_idx + i) % len(FALLBACK_MODELS)
        rotation.append(FALLBACK_MODELS[idx])
        
    return rotation

def list_groq_models() -> list[dict]:
    try:
        models = client.models.list()
        # models.data es una lista de objetos Model
        return [{"id": m.id, "owned_by": m.owned_by} for m in models.data if "llama" in m.id.lower() or "mixtral" in m.id.lower()]
    except Exception as e:
        logger.error(f"[GroqClient] Error listando modelos: {e}")
        return [{"id": m, "owned_by": "fallback"} for m in FALLBACK_MODELS]

def analyze_with_groq(system_prompt: str, user_prompt: str, groq_model: str = "llama-3.1-8b-instant") -> dict:
    models_to_try = get_models_to_try(groq_model)
    
    for i, model in enumerate(models_to_try):
        try:
            logger.info(f"[GroqClient] Iniciando análisis con {model}...")
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model,
                response_format={"type": "json_object"},
                timeout=120,
                max_tokens=2000
            )
            response_text = chat_completion.choices[0].message.content
            parsed = json.loads(response_text)
            parsed["actual_model_used"] = model
            return parsed
        except RateLimitError as e:
            logger.warning(f"[GroqClient] RateLimitError con {model}: {e}. Intentando fallback...")
            if i == len(models_to_try) - 1:
                raise e # Ya no hay más fallbacks
            continue
        except Exception as e:
            logger.error(f"[GroqClient] Error al conectar con Groq ({model}): {e}")
            raise e

def generate_text_with_groq(system_prompt: str, user_prompt: str, groq_model: str = "llama-3.1-8b-instant") -> str:
    """Llama a Groq y devuelve texto plano (sin JSON). Útil para generar nombres cortos."""
    models_to_try = get_models_to_try(groq_model)
    
    for i, model in enumerate(models_to_try):
        try:
            logger.info(f"[GroqClient] Generando texto con {model}...")
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=model,
                timeout=120,
                max_tokens=2000
            )
            text = chat_completion.choices[0].message.content.strip()
            # En texto plano, no podemos inyectar un JSON, pero podemos usar un delimitador o simplemente confiar en analyze_with_groq
            return text
        except RateLimitError as e:
            logger.warning(f"[GroqClient] RateLimitError con {model}: {e}. Intentando fallback...")
            if i == len(models_to_try) - 1:
                raise e
            continue
        except Exception as e:
            logger.error(f"[GroqClient] Error generando texto ({model}): {e}")
            raise e

def chat_with_groq(messages: list[dict], temperature: float = 0.7, groq_model: str = "llama-3.1-8b-instant") -> str:
    models_to_try = get_models_to_try(groq_model)
    for i, model in enumerate(models_to_try):
        try:
            logger.info(f"[GroqClient] Iniciando chat con {model}...")
            chat_completion = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                timeout=120,
                max_tokens=2000
            )
            return chat_completion.choices[0].message.content
        except RateLimitError as e:
            logger.warning(f"[GroqClient] RateLimitError con {model}: {e}. Intentando fallback...")
            if i == len(models_to_try) - 1:
                raise e
            continue
        except Exception as e:
            logger.error(f"[GroqClient] Error en chat ({model}): {e}")
            raise e
