import os
import json
import logging
from groq import Groq

logger = logging.getLogger(__name__)

# Configuración del cliente Groq
# Usará automáticamente la variable de entorno GROQ_API_KEY
client = Groq()

def analyze_with_groq(prompt: str) -> dict:
    try:
        logger.info("[GroqClient] Iniciando análisis con llama-3.3-70b-versatile...")
        # Timeout de 2 segundos según lo solicitado (Groq es rápido, 2s para failover rápido)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior academic reviewer. Always respond in valid JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            timeout=30.0
        )

        response_text = chat_completion.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        logger.error(f"[GroqClient] Error al conectar con Groq: {e}")
        raise e
