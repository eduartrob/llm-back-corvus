from pydantic import BaseModel
from typing import Optional

# analyzeproposal

class AnalyzeProposalRequest(BaseModel):
    proposal_text: str
    similar_projects: list[dict] = []
    max_sim_pct: float = 0.0
    risk_level: str = "Bajo"
    project_name: str = "NUEVA_PROPUESTA"
    top_project_name: str = "Ninguno"
    provider: str = "ollama"

# sessionstart

class StartSessionRequest(BaseModel):
    user_id: str
    proposal_summary: str
    analysis_result: dict

class StartSessionResponse(BaseModel):
    session_id: str
    mode: str               # 'defense' | 'rejection'
    ai_opening_message: str
    quota: Optional[dict] = None

# sessionmessage

class SessionMessageRequest(BaseModel):
    session_id: str
    user_message: str

class SessionMessageResponse(BaseModel):
    ai_message: str
    session_id: str

class GenerateNameRequest(BaseModel):
    prompt: str
    provider: str = "ollama"

class GenerateRAGSummaryRequest(BaseModel):
    query: str
    context: str
    provider: str = "groq"
