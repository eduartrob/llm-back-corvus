import os
import json
import logging
from groq import Groq

logger = logging.getLogger(__name__)

from app.config import settings

# Configuración del cliente Groq
client = Groq(api_key=settings.GROQ_API_KEY)

def analyze_with_groq(system_prompt: str, user_prompt: str) -> dict:
    try:
        logger.info("[GroqClient] Iniciando análisis con llama-3.3-70b-versatile...")
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            timeout=None,
            max_tokens=4000
        )

        response_text = chat_completion.choices[0].message.content
        return json.loads(response_text)
    except Exception as e:
        logger.error(f"[GroqClient] Error al conectar con Groq: {e}")
        raise e

def generate_text_with_groq(system_prompt: str, user_prompt: str) -> str:
    """Llama a Groq y devuelve texto plano (sin JSON). Útil para generar nombres cortos."""
    try:
        logger.info("[GroqClient] Generando texto con llama-3.3-70b-versatile...")
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            timeout=None,
            max_tokens=4000
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"[GroqClient] Error generando texto: {e}")
        raise e


def chat_with_groq(messages: list[dict], temperature: float = 0.7) -> str:
    try:
        logger.info("[GroqClient] Iniciando chat con llama-3.3-70b-versatile...")
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            timeout=None,
            max_tokens=2000
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"[GroqClient] Error en chat: {e}")
        raise e
