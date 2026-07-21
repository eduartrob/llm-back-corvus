with open("app/api/models.py", "r") as f:
    content = f.read()

# Change default to llama-3.1-8b-instant if not already
content = content.replace('groq_model: Optional[str] = "llama-3.3-70b-versatile"', 'groq_model: Optional[str] = "llama-3.1-8b-instant"')

with open("app/api/models.py", "w") as f:
    f.write(content)
