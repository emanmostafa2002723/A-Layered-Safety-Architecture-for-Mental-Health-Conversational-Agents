"""
LangGraph Pipeline State — SHARED CONTRACT
Owner: Lamya (P4)
Everyone must use exactly these field names.
Do not rename fields without team agreement.
"""
from typing import TypedDict, Annotated, List, Optional
import operator


class AgentState(TypedDict):
    # Input
    user_message: str
    clean_message: str       # after PII redaction
    user_id: str
    session_id: str

    # G1 results
    crisis_detected: bool
    crisis_confidence: float
    crisis_category: str
    boundary_violation_input: bool
    pii_detected: bool
    injection_detected: bool

    # LLM
    llm_response: Optional[str]
    llm_skipped: bool        # True when crisis bypass fires (confidence > 0.75)

    # G2 results
    hallucination_detected: bool
    boundary_violation_output: bool
    empathy_score: float
    regeneration_count: int

    # G3 results
    escalation_triggered: bool
    human_review_flagged: bool
    final_response: Optional[str]

    # Audit log — operator.add makes this append-only across nodes
    audit_log: Annotated[List[dict], operator.add]
