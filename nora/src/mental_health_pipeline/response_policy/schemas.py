"""Versioned, content-free contracts for response-policy orchestration.

The types in this module deliberately cannot carry the raw user message,
classifier probabilities, rule spans, identity data, or precise location.  The
raw message belongs in the user role of the generation request; it is not
policy metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


POLICY_SCHEMA_VERSION = "response-policy/1.0"
POLICY_VERSION = "response-policy-2026-10-19-conditional-guidance"


class SystemRoute(str, Enum):
    NORMAL_CHAT = "normal_chat"
    MONITORED_SUPPORT = "monitored_support"
    CLINICAL_BOUNDARY_SAFE_COMPLETION = "clinical_boundary_safe_completion"
    UNSUPPORTED_SCOPE = "unsupported_scope"
    URGENT_MEDICAL_RESPONSE = "urgent_medical_response"
    CRISIS_RESPONSE = "crisis_response"


class BoundaryRoute(str, Enum):
    ALLOW = "allow"
    SAFE_COMPLETION = "safe_completion"
    URGENT_MEDICAL_ESCALATION = "urgent_medical_escalation"


class CrisisTier(str, Enum):
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


@dataclass(frozen=True, slots=True)
class PolicyStateUpdate:
    """Content-free observation offered to the bounded session-state store."""

    categories: tuple[str, ...] = ()
    risk_signal_turn: bool = False
    highest_turn_tier: CrisisTier = CrisisTier.NORMAL
    highest_turn_severity: Severity = Severity.NONE
    temporal_status: TemporalStatus = TemporalStatus.UNSPECIFIED
    direct_safety_question_requested: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "categories": list(self.categories),
            "risk_signal_turn": self.risk_signal_turn,
            "highest_turn_tier": self.highest_turn_tier.value,
            "highest_turn_severity": self.highest_turn_severity.value,
            "temporal_status": self.temporal_status.value,
            "direct_safety_question_requested": self.direct_safety_question_requested,
        }


@dataclass(frozen=True, slots=True)
class ResponsePolicyPacket:
    """Trusted policy metadata passed separately from user-controlled text.

    ``clinical_deployment_authorized`` is intentionally fixed to ``False`` for
    this evidence-limited implementation.  Deployment engineering can use the
    packet, but it must not be presented as clinical approval.
    """

    system_route: SystemRoute
    situation_codes: tuple[str, ...]
    crisis_labels: tuple[str, ...]
    rule_kinds: tuple[str, ...]
    suppressed_rule_kinds: tuple[str, ...]
    evidence_sources: Mapping[str, tuple[str, ...]]
    temporal_status: TemporalStatus
    severity: Severity
    crisis_tier: CrisisTier
    boundary_route: BoundaryRoute
    boundary_labels: tuple[str, ...]
    boundary_review_reasons: tuple[str, ...]
    topic_tags: tuple[str, ...]
    needs_clarification: bool
    generator_guidance_id: str
    secondary_guidance_ids: tuple[str, ...]
    tone: tuple[str, ...]
    response_goals: tuple[str, ...]
    prohibited_behaviors: tuple[str, ...]
    required_visible_elements: tuple[str, ...]
    generation_allowed: bool
    fixed_content_timing: str
    fallback_template_id: str
    output_guard_profile: str
    policy_flags: tuple[str, ...] = ()
    state_update: PolicyStateUpdate = field(default_factory=PolicyStateUpdate)
    schema_version: str = POLICY_SCHEMA_VERSION
    policy_version: str = POLICY_VERSION
    output_guard_required: bool = True
    delivery_mode: str = "buffer_full_response_until_output_guard_approval"
    clinical_deployment_authorized: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported policy schema: {self.schema_version!r}")
        if not self.policy_version:
            raise ValueError("policy_version must be non-empty")
        if len(self.response_goals) > 3:
            raise ValueError("A response policy may include at most three immediate goals")
        if not self.generator_guidance_id:
            raise ValueError("generator_guidance_id must be non-empty")
        if not self.fallback_template_id or not self.output_guard_profile:
            raise ValueError("fallback and output-guard profiles are required")
        if self.fixed_content_timing not in {"before_generation", "with_response"}:
            raise ValueError("Invalid fixed_content_timing")
        if self.delivery_mode != "buffer_full_response_until_output_guard_approval":
            raise ValueError("Partial or streaming delivery is not authorized")
        if not self.output_guard_required:
            raise ValueError("The output guard cannot be disabled by a policy packet")
        if self.clinical_deployment_authorized:
            raise ValueError("Current research evidence does not authorize clinical deployment")
        for values in (
            self.situation_codes,
            self.crisis_labels,
            self.rule_kinds,
            self.suppressed_rule_kinds,
            self.boundary_labels,
            self.boundary_review_reasons,
            self.topic_tags,
            self.response_goals,
            self.prohibited_behaviors,
            self.required_visible_elements,
            self.policy_flags,
        ):
            if len(values) != len(set(values)):
                raise ValueError("Policy tuple fields must be de-duplicated")

    @property
    def primary_situation_code(self) -> str | None:
        return self.situation_codes[0] if self.situation_codes else None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready policy data without raw text or probabilities."""

        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "system_route": self.system_route.value,
            "situation_codes": list(self.situation_codes),
            "crisis_labels": list(self.crisis_labels),
            "rule_kinds": list(self.rule_kinds),
            "suppressed_rule_kinds": list(self.suppressed_rule_kinds),
            "evidence_sources": {
                label: list(sources)
                for label, sources in sorted(self.evidence_sources.items())
            },
            "temporal_status": self.temporal_status.value,
            "severity": self.severity.value,
            "crisis_tier": self.crisis_tier.value,
            "boundary_route": self.boundary_route.value,
            "boundary_labels": list(self.boundary_labels),
            "boundary_review_reasons": list(self.boundary_review_reasons),
            "topic_tags": list(self.topic_tags),
            "needs_clarification": self.needs_clarification,
            "generator_guidance_id": self.generator_guidance_id,
            "secondary_guidance_ids": list(self.secondary_guidance_ids),
            "tone": list(self.tone),
            "response_goals": list(self.response_goals),
            "prohibited_behaviors": list(self.prohibited_behaviors),
            "required_visible_elements": list(self.required_visible_elements),
            "generation_allowed": self.generation_allowed,
            "fixed_content_timing": self.fixed_content_timing,
            "fallback_template_id": self.fallback_template_id,
            "output_guard_profile": self.output_guard_profile,
            "output_guard_required": self.output_guard_required,
            "delivery_mode": self.delivery_mode,
            "policy_flags": list(self.policy_flags),
            "state_update": self.state_update.to_dict(),
            "clinical_deployment_authorized": self.clinical_deployment_authorized,
        }


@dataclass(frozen=True, slots=True)
class PolicyBuildResult:
    """Policy plus an optional updated bounded state snapshot."""

    packet: ResponsePolicyPacket
    recent_state: Any | None = None

