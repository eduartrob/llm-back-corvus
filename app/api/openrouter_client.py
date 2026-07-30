import httpx
import json
import logging

logger = logging.getLogger(__name__)

OPENROUTER_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "deepseek/deepseek-r1:free",
    "openai/gpt-4o-mini"
]

async def generate_openrouter_json(system_prompt: str, user_prompt: str, api_key: str, model: str = "google/gemini-2.0-flash-exp:free") -> dict:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "HTTP-Referer": "https://corvus.eduartrob.site",
        "X-Title": "Corvus Platform",
        "Content-Type": "application/json",
    }
    
    models_to_try = [model] + [m for m in OPENROUTER_MODELS if m != model]
    last_err = None
    
    for attempt_model in models_to_try:
        payload = {
            "model": attempt_model,
            "messages": [
                {"role": "system", "content": system_prompt + "\n\nCRITICAL INSTRUCTION: Return ONLY a valid JSON object matching the requested schema. Do not add conversational text."},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        
        try:
            logger.info(f"[OpenRouterClient] Solicitando análisis a {attempt_model}...")
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, headers=headers, json=payload, timeout=90.0)
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                return json.loads(content)
        except Exception as e:
            logger.warning(f"[OpenRouterClient] Falló {attempt_model}: {e}")
            last_err = e
            continue

    raise last_err if last_err else Exception("No se pudo obtener respuesta de ningún modelo de OpenRouter")
