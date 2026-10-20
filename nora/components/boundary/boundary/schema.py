"""One contract for rules, NLI comparison, supervised inference and integration."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
import unicodedata

LABELS = ("diagnosis", "medication_decision", "clinical_treatment", "harmful_medical_instruction")
LEGACY_LABELS = {
    "diagnosis": "diagnosis_or_personal_interpretation",
    "medication_decision": "medication_or_personal_treatment",
    "clinical_treatment": "medication_or_personal_treatment",
    "harmful_medical_instruction": "harmful_or_unethical_medical_instruction",
}


def normalize(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Expected a nonempty message")
    return unicodedata.normalize("NFKC", text).replace("’", "'").replace("‘", "'").strip()


def text_key(text: str) -> str:
    return re.sub(r"[^\w]+", " ", normalize(text).casefold()).strip()


@dataclass
class Decision:
    decision_kind: str
    reasons: list[str] = field(default_factory=list)
    safety_signals: list[str] = field(default_factory=list)
    status: str = "ok"
    complete_input_seen: bool = True
    backend: str = "unknown"
    evidence: dict = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        value = asdict(self)
        urgent = "medical_emergency" in self.safety_signals
        legacy = sorted({LEGACY_LABELS[x] for x in self.reasons if x in LEGACY_LABELS})
        if urgent:
            legacy.append("urgent_medical_safety")
        value.update({
            "schema_version": "boundary-rebuild-1.0",
            "suggested_route": "urgent_medical_escalation" if urgent else
                "allow" if self.decision_kind == "allow" else "safe_completion",
            "labels": legacy,
            "needs_review": self.decision_kind in {"uncertain", "unavailable"},
            "input_coverage": {"complete_input_seen": self.complete_input_seen},
            "explicit_emergency_screen": {"complete": True, "explicit_signal": urgent},
            "clinical_deployment_ready": False,
            "scores_are_calibrated_clinical_probabilities": False,
            "requires_rebuild_routing_adapter": True,
        })
        return value

