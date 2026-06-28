from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_HOST: str = "http://ollama:11434"
    ALLOWED_MODEL: str = "llama3.2:1b"
    MAX_CONCURRENT_REQUESTS: int = 2
    UNLIMITED_SESSIONS: bool = True
    FREE_SESSION_LIMIT: int = 2
    AUTH_SERVICE_URL: str = "http://auth-service:3001"
    PORT: int = 3003

    class Config:
        env_file = ".env"

settings = Settings()
