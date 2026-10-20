"""Small, conservative retrieval-needed classifier for the RAG pipeline.

This module makes only the retrieval decision. Safety routing remains a separate,
higher-priority decision in the integrated application.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class RetrievalDecision:
    requires_retrieval: bool
    intent: str
    reason: str


_MENTAL_HEALTH = re.compile(
    r"\b(?:mental health|anxiet\w*|anxious|panic|stress\w*|depress\w*|ptsd|trauma\w*|"
    r"therap\w*|counsel\w*|psych\w*|schizophren\w*|bipolar|hallucinat\w*|"
    r"delusion\w*|antidepressant\w*|antipsychotic\w*|ssri\w*|medication\w*|"
    r"suicid\w*|self harm|self injur\w*|abuse\w*|neglect\w*|violence|"
    r"substance\w*|opioid\w*|cannabis|alcohol|addict\w*|sleep\w*|"
    r"coping|wellbeing)\b",
    re.IGNORECASE,
)

_SAFETY_OR_DISTRESS = re.compile(
    r"\b(?:want to die|kill myself|end my life|don t want to live|"
    r"hurt myself|harm myself|hopeless|worthless|overdose|"
    r"raped|sexual assault|my partner hits|my partner hurts)\b",
    re.IGNORECASE,
)

_SOCIAL = re.compile(
    r"(?:hi|hey|hello|hello there|hi there|good morning|good afternoon|"
    r"good evening|good night|thanks|thank you|thank you so much|"
    r"that was helpful thanks|that was helpful thank you|you re welcome|"
    r"okay i understand|ok i understand|got it|goodbye|bye|bye bye|"
    r"see you later|take care)",
    re.IGNORECASE,
)

_CLARIFY = re.compile(
    r"(?:i have a question|can we talk|i don t know what to say|"
    r"are you a real person|are you human|what can you help me with|"
    r"what do you do|how can you help me)",
    re.IGNORECASE,
)

_OUT_OF_SCOPE = (
    (re.compile(r"\b(?:weather|forecast)\b", re.IGNORECASE), "live_weather"),
    (re.compile(r"\b(?:football|soccer|sports? (?:score|result|match)|\bmatch\b)\b", re.IGNORECASE), "live_sports"),
    (re.compile(r"\b(?:bitcoin|cryptocurrency|stock price|exchange rate)\b", re.IGNORECASE), "live_finance"),
    (re.compile(r"\b(?:computer code|programming|code that|write (?:python|javascript|computer)|sorts? a list)\b", re.IGNORECASE), "programming"),
    (re.compile(r"\b(?:restaurant|nearby cafe|local dining)\b", re.IGNORECASE), "local_recommendation"),
)


def _normalize(message: str) -> str:
    text = unicodedata.normalize("NFKC", message).casefold()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def route_retrieval(message: str) -> RetrievalDecision:
    """Return a high-precision skip decision, defaulting uncertain inputs to RAG.

    The caller must run its safety guardrail first. In particular, an urgent
    safety route must never be replaced by this classifier's generic response.
    """
    if not isinstance(message, str):
        raise TypeError("message must be a string")
    text = _normalize(message)
    if not text:
        return RetrievalDecision(False, "clarify", "empty_message")

    # A substantive mental-health request wins over a greeting or an unrelated
    # keyword: "Hi, what is depression?" must still reach retrieval.
    if _SAFETY_OR_DISTRESS.search(text):
        return RetrievalDecision(True, "mental_health", "safety_or_distress_signal")
    if _MENTAL_HEALTH.search(text):
        return RetrievalDecision(True, "mental_health", "mental_health_signal")
    if _SOCIAL.fullmatch(text):
        return RetrievalDecision(False, "social", "social_message")
    if _CLARIFY.fullmatch(text):
        return RetrievalDecision(False, "clarify", "underspecified_or_identity_question")
    for pattern, reason in _OUT_OF_SCOPE:
        if pattern.search(text):
            return RetrievalDecision(False, "out_of_scope", reason)
    return RetrievalDecision(True, "uncertain", "conservative_default")


def respond_without_retrieval(message: str, decision: RetrievalDecision) -> str:
    """Give a short bounded response for a message deliberately skipped by RAG."""
    if decision.requires_retrieval:
        raise ValueError("This decision requires retrieval")
    text = _normalize(message)

    if decision.intent == "social":
        if text.startswith("good morning"):
            return "Good morning! How can I help?"
        if text.startswith("good afternoon"):
            return "Good afternoon! How can I help?"
        if text.startswith("good evening"):
            return "Good evening! How can I help?"
        if text.startswith(("goodbye", "bye", "see you later", "take care", "good night")):
            return "Goodbye. Take care."
        if text.startswith("that was helpful"):
            return "I'm glad it helped."
        if text.startswith(("thank", "thanks")):
            return "You're welcome."
        if text.startswith(("okay", "ok", "got it")):
            return "Okay. Let me know if you have another question."
        return "Hello! How can I help?"

    if decision.intent == "clarify":
        if "real person" in text or "are you human" in text:
            return "I'm an AI assistant, not a person. I can share mental-health information and help you think through questions."
        if "what can you help" in text or "what do you do" in text or "how can you help" in text:
            return "I can explain mental-health topics, offer general coping information, and help you find when professional support may be useful."
        if "don t know what to say" in text:
            return "That's okay. You can start anywhere, or tell me what you'd like help with."
        if "can we talk" in text:
            return "Yes. What's on your mind?"
        if "i have a question" in text:
            return "Of course—what would you like to ask?"
        return "What would you like help with?"

    if decision.reason == "live_weather":
        return "I don't have a live weather forecast. Please check a local weather service."
    if decision.reason == "live_sports":
        return "I don't have live sports results. Please check a sports results service."
    if decision.reason == "live_finance":
        return "I don't have live market prices. Please check a financial data service."
    if decision.reason == "programming":
        return "Programming is outside this mental-health assistant's scope."
    if decision.reason == "local_recommendation":
        return "I don't have local restaurant data. Please check a local listings or maps service."
    raise ValueError(f"No response for skip reason: {decision.reason}")
