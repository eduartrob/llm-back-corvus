from pydantic import BaseModel
from typing import Optional


# ─── analyze-proposal ───────────────────────────────────────────────────────

class AnalyzeProposalRequest(BaseModel):
    proposal_text: str
    similar_projects: list[dict] = []


# ─── session/start ──────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    user_id: str
    proposal_summary: str
    analysis_result: dict  # JSON completo de analyze-proposal


class StartSessionResponse(BaseModel):
    session_id: str
    mode: str               # 'defense' | 'rejection'
    ai_opening_message: str
    quota: Optional[dict] = None


# ─── session/message ────────────────────────────────────────────────────────

class SessionMessageRequest(BaseModel):
    session_id: str
    user_message: str


class SessionMessageResponse(BaseModel):
    ai_message: str
    session_id: str
