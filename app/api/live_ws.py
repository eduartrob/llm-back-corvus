import os
import json
import base64
import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import websockets

import httpx

router = APIRouter()
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HOST = "generativelanguage.googleapis.com"
MODEL = "models/gemini-2.0-flash-exp"
WS_URL = f"wss://{HOST}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={GEMINI_API_KEY}"

@router.websocket("/ws/defense-live/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    # Wait for the initial setup message from the client
    try:
        initial_msg = await websocket.receive_text()
        context_data = json.loads(initial_msg)
    except Exception as e:
        logger.error(f"Failed to receive setup from client: {e}")
        await websocket.close(code=1003)
        return
        
    student_name = context_data.get("student_name", "el alumno")
    summary = context_data.get("proposal_summary", "")
    email = context_data.get("email")

    # Validar suscripción activa en el microservicio de pagos
    if email:
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(f"http://payments-service:8001/pagos/suscripcion/{email}", timeout=5.0)
                if res.status_code == 200:
                    sub_data = res.json()
                    if not sub_data.get("activa"):
                        logger.warning(f"WebSocket rechazado: Suscripción inactiva para {email}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "Tu suscripción Plan Pro ha expirado o está inactiva."
                        }))
                        await websocket.close(code=4003)
                        return
        except Exception as e:
            logger.warning(f"No se pudo verificar la suscripción con payments-service: {e}")
    
    system_instruction = f"""
    Eres un estricto sinodal evaluador de tesis de la Universidad Politécnica de Chiapas.
    El alumno {student_name} está defendiendo el siguiente proyecto:
    {summary}
    Hazle preguntas difíciles, cortas y directas sobre su metodología, viabilidad y resultados.
    Tus respuestas deben ser breves, habladas con naturalidad, como en una conversación telefónica real.
    """

    setup_message = {
        "setup": {
            "model": MODEL,
            "generationConfig": {
                "responseModalities": ["AUDIO"]
            },
            "systemInstruction": {
                "parts": [{"text": system_instruction}]
            }
        }
    }

    try:
        async with websockets.connect(WS_URL) as gemini_ws:
            # Send setup to Gemini
            await gemini_ws.send(json.dumps(setup_message))
            
            # Wait for setup complete from Gemini
            setup_response = await gemini_ws.recv()
            logger.info(f"Gemini setup response: {setup_response}")

            # Notify flutter client we are ready
            await websocket.send_text(json.dumps({"type": "ready"}))

            async def receive_from_client():
                try:
                    while True:
                        # Client sends binary PCM audio chunks
                        data = await websocket.receive_bytes()
                        # Send to Gemini
                        payload = {
                            "realtimeInput": {
                                "mediaChunks": [{
                                    "mimeType": "audio/pcm;rate=16000",
                                    "data": base64.b64encode(data).decode('utf-8')
                                }]
                            }
                        }
                        await gemini_ws.send(json.dumps(payload))
                except WebSocketDisconnect:
                    logger.info("Client disconnected")
                except Exception as e:
                    logger.error(f"Error reading from client: {e}")

            async def receive_from_gemini():
                try:
                    while True:
                        msg = await gemini_ws.recv()
                        resp = json.loads(msg)
                        
                        if "serverContent" in resp:
                            model_turn = resp["serverContent"].get("modelTurn")
                            if model_turn:
                                parts = model_turn.get("parts", [])
                                for part in parts:
                                    if "inlineData" in part:
                                        b64_data = part["inlineData"].get("data")
                                        if b64_data:
                                            # Decode b64 to binary and send to flutter client
                                            audio_bytes = base64.b64decode(b64_data)
                                            await websocket.send_bytes(audio_bytes)
                                            
                                    # Forward text if available for captions
                                    if "text" in part:
                                        await websocket.send_text(json.dumps({
                                            "type": "text",
                                            "text": part["text"]
                                        }))
                                        
                        if "turnComplete" in resp.get("serverContent", {}):
                             await websocket.send_text(json.dumps({"type": "turnComplete"}))
                except websockets.exceptions.ConnectionClosed:
                    logger.info("Gemini connection closed")
                except Exception as e:
                    logger.error(f"Error receiving from Gemini: {e}")

            # Run both loops concurrently
            await asyncio.gather(
                receive_from_client(),
                receive_from_gemini()
            )

    except Exception as e:
        logger.error(f"WebSocket session failed: {e}")
        try:
            await websocket.close()
        except:
            pass
