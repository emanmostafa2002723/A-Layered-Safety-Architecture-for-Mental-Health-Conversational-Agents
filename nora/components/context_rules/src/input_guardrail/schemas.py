"""Typed records emitted by the input guardrail.

The public decision deliberately omits the raw message.  ``redacted_text`` is
safe for ordinary development logs, while production deployments should still
apply an explicit retention policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Tier(str, Enum):
    NORMAL = "normal"
    MONITORED = "monitored"
    CRISIS = "crisis"


class Severity(str, Enum):
    NONE = "none"
    WATCH = "watch"
    ELEVATED = "elevated"
    URGENT = "urgent"


class TemporalStatus(str, Enum):
    CURRENT = "current"
    IMMINENT = "imminent"
    RECENT = "recent"
    PAST = "past"
    MIXED = "mixed"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    text_view: str = "normalized"


@dataclass(frozen=True)
class SafetyMatch:
    rule_id: str
    category: str | None
    kind: str
    severity: Severity
    temporal: TemporalStatus
    span: Span
    suppressed: bool = False
    suppression_reason: str | None = None
    evidence_confidence: str = "high"


@dataclass(frozen=True)
class PiiMatch:
    pii_type: str
    start: int
    end: int


@dataclass
class SanitizationResult:
    message_id: str
    normalized_text: str
    redacted_text: str
    pii_matches: list[PiiMatch] = field(default_factory=list)
    injection_rule_ids: list[str] = field(default_factory=list)
    replacements: list[str] = field(default_factory=list)
    language_hint: str = "unknown"


@dataclass
class RuleExtraction:
    matches: list[SafetyMatch] = field(default_factory=list)
    topic_tags: list[str] = field(default_factory=list)
    global_temporal: TemporalStatus = TemporalStatus.UNSPECIFIED

    @property
    def active_matches(self) -> list[SafetyMatch]:
        return [match for match in self.matches if not match.suppressed]


@dataclass
class FusionEvent:
    rule_id: str
    outcome: str
    detail: str


@dataclass
class GuardrailDecision:
    schema_version: str
    message_id: str
    tier: Tier
    primary_category: str
    categories: list[str]
    temporal_status: TemporalStatus
    severity: Severity
    topic_tags: list[str]
    redacted_text: str
    classifier: dict[str, Any]
    rule_trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return _serialize(asdict(self))


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value
