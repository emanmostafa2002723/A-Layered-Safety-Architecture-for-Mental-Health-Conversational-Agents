"""Minimal routing/response contract. Does not call RAG or a text generator.

Use this adapter: old code ignores safety_signals and treats every uncertainty
as a diagnosis/medication request. Predict compatibility alone cannot fix that.
"""
from __future__ import annotations


def resolve(crisis: dict, boundary: dict) -> dict:
    signals = set(boundary.get("safety_signals", []))
    reasons = set(boundary.get("reasons", []))
    kind = boundary.get("decision_kind", "unavailable")
    if "medical_emergency" in signals:
        route, guidance = "urgent_medical_response", "urgent_medical"
    elif "child_safeguarding" in signals:
        route, guidance = "crisis_response", "child_safeguarding"
    elif "suicide_safety" in signals or crisis.get("tier") == "crisis":
        route, guidance = "crisis_response", "crisis_support"
    elif reasons:
        route, guidance = "clinical_boundary_safe_completion", "restricted_request"
    elif kind in {"uncertain", "unavailable"} or boundary.get("status") != "ok":
        route, guidance = "clinical_boundary_safe_completion", "clarify_or_general_support"
    elif crisis.get("tier") == "monitored":
        route, guidance = "monitored_support", "support_with_safety_check"
    else:
        route, guidance = "normal_chat", "ordinary_support_or_information"
    notices = []
    if "diagnosis" in reasons:
        notices.append("I can explain general information, but I cannot determine a diagnosis for someone.")
    if "medication_decision" in reasons:
        notices.append("I cannot choose a medicine, dose or medication change for someone.")
    if "clinical_treatment" in reasons:
        notices.append("I can discuss general options, but I cannot select your clinical treatment or replace your clinician.")
    if "harmful_medical_instruction" in reasons:
        notices.append("I cannot help carry out harmful or unsafe medical actions.")
    # A detected boundary request must not prepend a routine refusal to a
    # safeguarding/emergency response. Reasons remain in the boundary audit;
    # the higher-priority route determines what guidance is actually shown.
    if guidance != "restricted_request":
        notices = []
    return {
        "route":route, "guidance":guidance, "boundary_notices":notices,
        "safety_signals":sorted(signals), "crisis_signal_retained":crisis.get("tier") in {"crisis","monitored"},
        "prohibit_personalized_diagnosis":True, "prohibit_medication_decisions":True,
        "allow_general_support":True, "allow_knowledge_retrieval":route=="normal_chat",
        "requires_emergency_resources":bool(signals) or crisis.get("tier")=="crisis",
        "uncertainty_is_not_a_medical_request":guidance=="clarify_or_general_support",
    }

