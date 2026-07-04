import asyncio
from app.api.routes import build_groq_analysis_prompt
from app.api.groq_client import analyze_with_groq
import json

# Simulate the student's proposal
proposal_text = """
CONTEXTO DE LA PROBLEMÁTICA 
En Chiapas, las búsquedas de personas desaparecidas suelen realizarse en zonas rurales...
PROBLEMÁTICA 
La falta de un método para diseñar las trayectorias específicas...
NOMBRE LARGO 
Sistema inteligente para la optimización de rutas de patrullaje y rescate mediante algoritmos genéticos 
NOMBRE CORTO 
BusqueZone 
DESCRIPCIÓN DEL PROYECTO 
El proyecto consiste en diseñar un sistema de optimización...
VARIABLES DE DECISIÓN 
◦ Puntos de paso 
VARIABLES A OPTIMIZAR 
1. Tiempo total de búsqueda 
OBJETIVOS DE OPTIMIZACIÓN 
I. Minimizar el tiempo total de búsqueda 
BASE DE CONOCIMIENTO 
■ Velocidad promedio de desplazamiento
ENTRADAS AL SISTEMA 
• Número total de equipos
SALIDAS ESPERADAS DEL SISTEMA 
✔ Tabla con el camino completo
"""

system_prompt, user_prompt = build_groq_analysis_prompt(
    proposal_text=proposal_text,
    context_text="No hay proyectos similares previos. Es 100% original.",
    project_name="BusqueZone",
    top_project_name="Ninguno",
    max_sim_pct=0.0,
    risk_level="Bajo"
)

try:
    result = analyze_with_groq(system_prompt, user_prompt)
    print(json.dumps(result, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Error: {e}")
