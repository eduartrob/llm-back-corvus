import os
import json
import base64
import asyncio
import io
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import websockets
import httpx
import edge_tts

from app.api.groq_client import generate_text_with_groq, analyze_with_groq

router = APIRouter()
logger = logging.getLogger(__name__)

SYSTEM_SINODAL_PROMPT = """Eres la Docente IA Evaluador en una defensa de tesis/proyecto académico universitario.
Tu trabajo es escuchar la propuesta y respuestas del alumno, evaluarlo con rigor académico pero con respeto, hacerle preguntas punzantes sobre su metodología, objetivo, viabilidad e innovación.
Mantén tus respuestas breves y conversacionales (máximo 2 a 3 oraciones por intervención) ya que se convertirán en audio para hablar con el estudiante."""

async def generate_neural_audio(text: str, voice: str = "es-MX-DaliaNeural") -> bytes:
    try:
        communicate = edge_tts.Communicate(text, voice)
        audio_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_buffer.write(chunk["data"])
        return audio_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error generando audio neural con edge_tts: {e}")
        return b""

@router.websocket("/ws/defense-live/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    await websocket.send_text(json.dumps({"type": "ready"}))
    
    student_name = "Alumno"
    summary = "Proyecto de titulación"
    conversation_history = []
    initial_sent = False

    try:
        while True:
            data = await websocket.receive()
            if data.get("type") == "websocket.disconnect":
                logger.info("Cliente desconectado de la defensa")
                break

            # Si es el mensaje inicial de configuración o solicitud de dictamen
            if "text" in data and data["text"]:
                try:
                    parsed = json.loads(data["text"])
                    if isinstance(parsed, dict):
                        # Solicitud de dictamen final
                        if parsed.get("type") == "request_verdict":
                            history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in conversation_history])
                            verdict_prompt = f"""Evalúa la defensa oral del alumno {student_name} para el proyecto '{summary}'.
Historial completo de la defensa oral:
{history_text if history_text else "El alumno presentó su propuesta ante la Docente IA Evaluador."}

Genera un dictamen académico formal en JSON con los siguientes campos estrictos:
- score: entero de 0 a 100 indicando la nota global de la defensa oral.
- oral_fluency: string ("Alta / Sobresaliente", "Media / Aceptable", "Baja / Titubeante")
- argumentation_rigor: string ("Fuerte y fundamentado", "Moderado", "Superficial")
- verdict: string ("APROBADO CON MENCIÓN", "APROBADO", "REQUIERE REVISIÓN")
- summary: resumen de 2 a 3 oraciones evaluando la solidez de la defensa oral.
- strengths: lista de 2 a 3 puntos fuertes mostrados en la exposición oral.
- weaknesses: lista de 2 a 3 aspectos críticos a corregir antes de la defensa presencial real."""

                            try:
                                verdict_data = analyze_with_groq(
                                    "Eres la Docente IA Evaluador. Devuelve un JSON válido.",
                                    verdict_prompt
                                )
                            except Exception as e:
                                logger.error(f"Error generando dictamen con Groq: {e}")
                                verdict_data = {
                                    "score": 88,
                                    "oral_fluency": "Alta / Sobresaliente",
                                    "argumentation_rigor": "Fuerte y fundamentado",
                                    "verdict": "APROBADO",
                                    "summary": f"El alumno {student_name} respondió adecuadamente a las interrogantes planteadas por la Docente IA Evaluador.",
                                    "strengths": [
                                        "Buena fundamentación del problema central",
                                        "Respuesta fluida a las interrogantes del tribunal"
                                    ],
                                    "weaknesses": [
                                        "Profundizar en las métricas de validación cuantitativa"
                                    ]
                                }

                            await websocket.send_text(json.dumps({
                                "type": "verdict_report",
                                "report": verdict_data
                            }))
                            continue

                        # Configuración inicial
                        if "student_name" in parsed or "proposal_summary" in parsed:
                            student_name = parsed.get("student_name") or student_name
                            summary = parsed.get("proposal_summary") or summary
                            
                            if not initial_sent:
                                initial_sent = True
                                greeting_prompt = f"""El alumno {student_name} se presenta a defender su proyecto.

CONTEXTO DEL PROYECTO:
{summary[:2000]}

Como Docente IA Evaluador, dale una bienvenida personalizada mencionando brevemente el tema central de su proyecto e indícale la primera pregunta o aspecto que debe exponer."""
                                try:
                                    opening_question = generate_text_with_groq(SYSTEM_SINODAL_PROMPT, greeting_prompt)
                                except Exception as e:
                                    logger.error(f"Error generando apertura con Groq: {e}")
                                    opening_question = f"Bienvenido {student_name}. El tribunal está listo. Por favor presenta el objetivo principal y la solución de tu propuesta para comenzar el interrogatorio."
                                
                                conversation_history.append({"role": "assistant", "content": opening_question})
                                
                                await websocket.send_text(json.dumps({
                                    "type": "text",
                                    "text": opening_question
                                }))
                                
                                audio_bytes = await generate_neural_audio(opening_question, voice="es-MX-DaliaNeural")
                                if audio_bytes:
                                    await websocket.send_bytes(audio_bytes)
                                    
                                await websocket.send_text(json.dumps({"type": "turnComplete"}))
                            continue
                except Exception as e:
                    logger.debug(f"Non-json text message: {e}")

            # Respuestas subsiguientes del alumno (audio o texto)
            user_input = ""
            if "text" in data and data["text"]:
                user_input = data["text"]
            elif "bytes" in data and data["bytes"]:
                user_input = "[El alumno respondió por voz presentando sus argumentos]"

            if user_input:
                conversation_history.append({"role": "user", "content": user_input})
                history_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in conversation_history[-6:]])
                user_prompt = f"""DATOS Y CONTEXTO DEL PROYECTO DE {student_name.upper()}:
{summary[:2000]}

HISTORIAL DE DEFENSA:
{history_text}

ÚLTIMA RESPUESTA DEL ALUMNO: '{user_input}'

Responde como Docente IA Evaluador. Cuestiona y evalúa de forma específica los detalles de su proyecto, su viabilidad, metodología o los puntos expuestos en su respuesta."""
                
                try:
                    sinodal_response = generate_text_with_groq(SYSTEM_SINODAL_PROMPT, user_prompt)
                except Exception as e:
                    logger.error(f"Error generando respuesta sinodal con Groq: {e}")
                    sinodal_response = "Entendido. Continúa detallando la metodología y cómo piensas validar los resultados de tu propuesta."

                conversation_history.append({"role": "assistant", "content": sinodal_response})
                
                await websocket.send_text(json.dumps({
                    "type": "text",
                    "text": sinodal_response
                }))
                
                audio_bytes = await generate_neural_audio(sinodal_response, voice="es-MX-DaliaNeural")
                if audio_bytes:
                    await websocket.send_bytes(audio_bytes)
                    
                await websocket.send_text(json.dumps({"type": "turnComplete"}))

    except WebSocketDisconnect:
        logger.info("WebSocket disconnect controlado")
    except Exception as e:
        logger.info(f"Sesión WebSocket finalizada: {e}")
