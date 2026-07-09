## [1.2.13] - 2026-07-09
- Preparación para nueva arquitectura de Proyectos Integradores.

# Changelog - LLM Backend Corvus

## [1.1.0] - 2026-07-02
### Refactorización y Prompts
- **Prompts Centralizados**: Se extrajo toda la lógica de los prompts (incluyendo el análisis Blue Ocean) del controlador principal y se centralizó en `app/core/prompts.py` para cumplir con el principio de responsabilidad única.
- **Inyección Dinámica de Valores**: Se eliminaron las respuestas mockeadas en el endpoint `/analyze-blue-ocean`. El LLM ahora usa los valores matemáticos calculados por Qdrant/K-Means (`max_sim_pct` y `risk_level`) inyectados en el system prompt para justificar la calificación de originalidad de manera determinista en lugar de alucinar valores.
- **Fail-Fast de IA**: Se implementó el manejo estricto de caídas del modelo, retornando HTTP 503 en lugar de fallbacks quemados, para asegurar que toda evaluación provenga exclusivamente de Llama 3.3.
- **Corrección Groq Client**: Se corrigió el cliente de Groq para instanciarse pasando explícitamente `settings.GROQ_API_KEY`, resolviendo los fallos de autenticación al inicializar la app.

## [1.0.0] - Versión Inicial
- Implementación de inferencia con Llama 3 a través del Groq Client.
- Endpoints de procesamiento semántico con Ollama.
