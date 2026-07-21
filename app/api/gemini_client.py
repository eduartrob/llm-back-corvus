import httpx
import logging

logger = logging.getLogger(__name__)

async def generate_text_with_gemini(system_prompt: str, user_prompt: str, api_key: str, model: str = "gemini-2.5-flash") -> str:
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
