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
    team_id: str
    team_members: Optional[list[str]] = None
    proposal_summary: str
    analysis_result: dict

class StartSessionResponse(BaseModel):
    session_id: str
    mode: str               # 'defense' | 'rejection'
    ai_opening_message: str
    messages: Optional[list[dict]] = None
    quota: Optional[dict] = None

# sessionmessage

class SessionMessageRequest(BaseModel):
    session_id: str
    user_message: str
    student_name: Optional[str] = None

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

class AnalyzeHomeworkRequest(BaseModel):
    title: str
    full_text: str
    provider: str = "groq"

class DetectedTechnology(BaseModel):
    tecnologia: str
    score: float

class AnalyzeHomeworkResponse(BaseModel):
    tecnologias_detectadas: list[DetectedTechnology]
    es_ia: Optional[bool] = None
    probabilidad_ia: Optional[float] = None

class DocumentItem(BaseModel):
    id: str
    name: str
    folder: str

class FilterSoftwareRequest(BaseModel):
    documents: list[DocumentItem]
    provider: str = "ollama"

class FilterSoftwareResponse(BaseModel):
    valid_ids: list[str]

class GenerateCareerSkillsRequest(BaseModel):
    career_name: str
    provider: str = "groq"

class CareerSkillItem(BaseModel):
    name: str
    weight: int

class GenerateCareerSkillsResponse(BaseModel):
    skills: list[CareerSkillItem]

class ValidateIdeaQuickRequest(BaseModel):
    idea: str
    blocked_topics: list[str] = []
    blocked_techs: list[str] = []
    similar_projects: list[dict] = []
    provider: str = "ollama"
