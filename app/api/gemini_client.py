import httpx
import logging

logger = logging.getLogger(__name__)

async def generate_text_with_gemini(system_prompt: str, user_prompt: str, api_key: str, model: str = "gemini-2.0-flash") -> str:
    """
    Genera texto usando la API REST nativa de Gemini.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "system_instruction": {
            "parts": [
                {
                    "text": system_prompt
                }
            ]
        },
        "contents": [
            {
                "parts": [
                    {
                        "text": user_prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 100
        }
    }

    try:
        logger.info(f"[GeminiClient] Generando texto con Gemini {model}...")
        async with httpx.AsyncClient() as client:
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            
            data = response.json()
            if "candidates" in data and len(data["candidates"]) > 0:
                parts = data["candidates"][0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
            
            raise Exception("Respuesta de Gemini no contiene candidatos válidos.")
    except Exception as e:
        logger.error(f"[GeminiClient] Error conectando a Gemini: {e}")
        if hasattr(e, 'response') and e.response is not None:
             logger.error(f"Detalles: {e.response.text}")
        raise e

async def generate_gemini_json(system_prompt: str, user_prompt: str, api_key: str, model: str = "gemini-2.0-flash") -> dict:
    """
    Genera JSON usando la API REST nativa de Gemini con response_mime_type.
    Intenta con el modelo principal, luego con gemini-2.0-flash-lite si hay rate limit.
    """
    import asyncio
    import json as json_lib

    GEMINI_MODELS = [model, "gemini-2.0-flash-lite"]
    last_error = None

    for attempt_model in GEMINI_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{attempt_model}:generateContent?key={api_key}"

        headers = {"Content-Type": "application/json"}

        payload = {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {"parts": [{"text": user_prompt}]}
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
                "response_mime_type": "application/json"
            }
        }

        try:
            logger.info(f"[GeminiClient] Generando JSON estructurado con Gemini {attempt_model}...")
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=payload, timeout=90.0)
                response.raise_for_status()

                data = response.json()
                if "candidates" in data and len(data["candidates"]) > 0:
                    parts = data["candidates"][0].get("content", {}).get("parts", [])
                    if parts:
                        raw_text = parts[0].get("text", "").strip()
                        return json_lib.loads(raw_text)

                raise Exception("Respuesta de Gemini no contiene candidatos válidos.")

        except Exception as e:
            last_error = e
            # If rate limited, wait a moment before trying the next model
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"[GeminiClient] {attempt_model} falló: {e.response.text[:200]}")
                if e.response.status_code == 429:
                    logger.warning(f"[GeminiClient] Rate limit en {attempt_model}, esperando 5s antes de intentar modelo alternativo...")
                    await asyncio.sleep(5)
                    continue
            else:
                logger.error(f"[GeminiClient] {attempt_model} error: {e}")
            break

    raise last_error
