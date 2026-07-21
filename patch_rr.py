with open("app/api/groq_client.py", "r") as f:
    content = f.read()

target = """FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]"""

replacement = """FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]

import threading
_rr_lock = threading.Lock()
_rr_index = 0

def get_models_to_try(requested_model: str) -> list[str]:
    global _rr_index
    
    # Si el modelo solicitado NO está en la lista pesada (ej: llama-3.1-8b-instant para chat),
    # lo intentamos primero, y si falla usamos la lista pesada como respaldo normal.
    if requested_model not in FALLBACK_MODELS:
        return [requested_model] + [m for m in FALLBACK_MODELS]
        
    # Si es uno de los modelos pesados, hacemos Round-Robin (1, 2, 3, 1, 2, 3...)
    with _rr_lock:
        start_idx = _rr_index
        _rr_index = (_rr_index + 1) % len(FALLBACK_MODELS)
        
    rotation = []
    for i in range(len(FALLBACK_MODELS)):
        idx = (start_idx + i) % len(FALLBACK_MODELS)
        rotation.append(FALLBACK_MODELS[idx])
        
    return rotation"""

content = content.replace(target, replacement)

# Now replace the occurrences of models_to_try assignment
target_assign = 'models_to_try = [groq_model] + [m for m in FALLBACK_MODELS if m != groq_model]'
replacement_assign = 'models_to_try = get_models_to_try(groq_model)'

content = content.replace(target_assign, replacement_assign)

with open("app/api/groq_client.py", "w") as f:
    f.write(content)
