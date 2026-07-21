with open("app/api/groq_client.py", "r") as f:
    content = f.read()

target = """FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768"
]"""

replacement = """FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b"
]"""

content = content.replace(target, replacement)

with open("app/api/groq_client.py", "w") as f:
    f.write(content)
