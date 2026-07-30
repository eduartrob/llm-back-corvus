# LLM AI Service — Corvus Platform

Este microservicio pertenece a la plataforma **CORVUS**. Es el motor central de **Inteligencia Artificial Generativa y Evaluación Académica** que impulsa la revisión automatizada de propuestas y la defensa en tiempo real.

---

## 🎯 Función en el Ecosistema CORVUS
* **Revisión de Propuestas (PDF):** Extracción e inspección automática de documentos PDF enviados por los equipos.
* **Evaluación Multimodal:** Análisis semántico con Ollama local (Llama 3.2 3B) y GroqCloud en la nube (Llama 3.3 70B Versatile).
* **Defensa Interactiva en Tiempo Real:** WebSocket streaming para simulaciones de defensa oral con jurado de IA.
* **Base de Datos Dedicada:** Opera sobre su base de datos PostgreSQL aislada **`corvus_llm_db`** (tabla `final_reviews`).

---

## ⚙️ Tecnologías
* **Lenguaje & Framework:** Python 3.10+, FastAPI, WebSockets.
* **Modelos IA:** Ollama (Llama 3.2 3B), Groq API (Llama 3.3 70B Versatile), OpenRouter API.
* **ORM:** SQLAlchemy.
* **Base de Datos:** PostgreSQL (`corvus_llm_db`).

---

## 🛠️ Ejecución Local Independiente

### 1. Entorno Virtual & Dependencias
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Variables de Entorno
Crea un archivo `.env` basado en `.env.example`:
```env
PORT=8005
DATABASE_URL="postgresql://corvus_user:password@localhost:5432/corvus_llm_db"
GROQ_API_KEY="your_groq_api_key"
OLLAMA_BASE_URL="http://localhost:11434"
```

### 3. Iniciar Servidor en Desarrollo
```bash
uvicorn app.main:app --reload --port 8005
```
Documentación OpenAPI disponible en `http://localhost:8005/docs`.

---

## 🐳 Ejecución con Docker

```bash
docker build -t corvus-llm-service .
docker run -p 8005:8005 --env-file .env corvus-llm-service
```

---

## 🔗 Integración con la Orquestación de CORVUS
Administrado centralmente por **`orchestration-back-corvus`** y accesible mediante los endpoints `/api/v1/llm` del API Gateway.
