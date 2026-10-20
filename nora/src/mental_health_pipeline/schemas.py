"""Versioned contracts shared by the application-level pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping
import re
from .personal_memory import MemoryNote


MAX_MESSAGE_BYTES = 16_384
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ContractError(ValueError):
    """A component returned a value that violates its public contract."""


class RequestValidationError(ValueError):
    """A caller supplied an invalid request shape."""


class SystemRoute(str, Enum):
    CRISIS = "crisis_response"
    URGENT_MEDICAL = "urgent_medical_response"
    BOUNDARY_SAFE_COMPLETION = "clinical_boundary_safe_completion"
    MONITORED = "monitored_support"
    NORMAL = "normal_chat"
    UNSUPPORTED_SCOPE = "unsupported_scope"


@dataclass(frozen=True)
class RequestScope:
    """Product metadata kept outside classifier payloads."""

    age_eligibility_18_plus: bool | None = None
    preferred_language: str = "en"
    current_location_country: str | None = None
    current_location_region: str | None = None
    location_confirmed: bool = False

    def normalized_country(self) -> str | None:
        if self.current_location_country is None:
            return None
        value = self.current_location_country.strip().upper()
        return value or None

    @property
    def is_supported(self) -> bool:
        return self.age_eligibility_18_plus is True and self.preferred_language.lower() == "en"

    @property
    def unsupported_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.age_eligibility_18_plus is not True:
            reasons.append("adult_eligibility_not_confirmed")
        if self.preferred_language.lower() != "en":
            reasons.append("unsupported_language")
        return tuple(reasons)


@dataclass(frozen=True)
class PipelineRequest:
    latest_user_message: str
    session_id: str
    scope: RequestScope
    turn_number: int = 1
    request_id: str | None = None
    personal_memory: tuple[MemoryNote, ...] = ()

    schema_version: str = field(default="pipeline-request/1.0", init=False)

    def validate(self) -> None:
        if (not isinstance(self.personal_memory, tuple) or len(self.personal_memory) > 3
                or any(not isinstance(n, MemoryNote) for n in self.personal_memory)):
            raise RequestValidationError('invalid personal memory context')
        if not isinstance(self.latest_user_message, str) or not self.latest_user_message.strip():
            raise RequestValidationError("latest_user_message must be a non-empty string")
        try:
            size = len(self.latest_user_message.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as error:
            raise RequestValidationError("latest_user_message must be valid UTF-8") from error
        if size > MAX_MESSAGE_BYTES:
            raise RequestValidationError(
                f"latest_user_message exceeds the {MAX_MESSAGE_BYTES}-byte limit"
            )
        if not isinstance(self.session_id, str) or not SESSION_ID_RE.fullmatch(self.session_id):
            raise RequestValidationError("session_id has an invalid format")
        if not isinstance(self.turn_number, int) or isinstance(self.turn_number, bool) or self.turn_number < 1:
            raise RequestValidationError("turn_number must be a positive integer")


VALID_CRISIS_TIERS = {"normal", "monitored", "crisis"}
VALID_TEMPORAL = {"current", "imminent", "recent", "past", "mixed", "unspecified"}
VALID_SEVERITY = {"none", "watch", "elevated", "urgent"}


@dataclass(frozen=True)
class CrisisDecision:
    tier: str
    primary_category: str
    categories: tuple[str, ...]
    temporal_status: str
    severity: str
    topic_tags: tuple[str, ...]
    classifier_labels: tuple[str, ...]
    near_threshold_labels: tuple[str, ...]
    rule_kinds: tuple[str, ...]
    evidence_sources: Mapping[str, tuple[str, ...]]
    status: str
    truncated: bool
    raw: Mapping[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CrisisDecision":
        if not isinstance(value, Mapping):
            raise ContractError("crisis decision must be a mapping")
        tier = value.get("tier")
        temporal = value.get("temporal_status", "unspecified")
        severity = value.get("severity", "none")
        if tier not in VALID_CRISIS_TIERS:
            raise ContractError(f"invalid crisis tier: {tier!r}")
        if temporal not in VALID_TEMPORAL:
            raise ContractError(f"invalid temporal status: {temporal!r}")
        if severity not in VALID_SEVERITY:
            raise ContractError(f"invalid crisis severity: {severity!r}")

        classifier = value.get("classifier") or {}
        if not isinstance(classifier, Mapping):
            raise ContractError("crisis classifier field must be a mapping")
        predicted = _string_tuple(classifier.get("predicted_labels", ()), "predicted_labels")
        near = _string_tuple(classifier.get("near_threshold_labels", ()), "near_threshold_labels")
        categories = _string_tuple(value.get("categories", ()), "categories")
        topic_tags = _string_tuple(value.get("topic_tags", ()), "topic_tags")

        trace = value.get("rule_trace") or {}
        channel = trace.get("rule_channel", {}) if isinstance(trace, Mapping) else {}
        matches = channel.get("matches", ()) if isinstance(channel, Mapping) else ()
        active_matches: list[Mapping[str, Any]] = []
        if isinstance(matches, list):
            active_matches = [
                item for item in matches
                if isinstance(item, Mapping) and not bool(item.get("suppressed"))
            ]
        rule_kinds = tuple(dict.fromkeys(
            str(item["kind"]) for item in active_matches if isinstance(item.get("kind"), str)
        ))
        evidence: dict[str, tuple[str, ...]] = {}
        for category in categories:
            sources: list[str] = []
            if category in predicted:
                sources.append("classifier")
            if any(item.get("category") == category for item in active_matches):
                sources.append("rules")
            evidence[category] = tuple(sources or ("unspecified",))

        return cls(
            tier=tier,
            primary_category=str(value.get("primary_category", "no_crisis")),
            categories=categories,
            temporal_status=temporal,
            severity=severity,
            topic_tags=topic_tags,
            classifier_labels=predicted,
            near_threshold_labels=near,
            rule_kinds=rule_kinds,
            evidence_sources=evidence,
            status=str(classifier.get("status", "unknown")),
            truncated=bool(classifier.get("truncated", False)),
            raw=value,
        )

    @classmethod
    def fail_closed(cls, status: str = "component_error") -> "CrisisDecision":
        raw = {
            "tier": "monitored",
            "primary_category": "no_crisis",
            "categories": [],
            "temporal_status": "unspecified",
            "severity": "watch",
            "topic_tags": [],
            "classifier": {"status": status, "predicted_labels": [], "near_threshold_labels": []},
            "rule_trace": {"rule_channel": {"matches": []}},
        }
        return cls.from_mapping(raw)


VALID_BOUNDARY_ROUTES = {"allow", "safe_completion", "urgent_medical_escalation"}


@dataclass(frozen=True)
class BoundaryDecision:
    suggested_route: str
    status: str
    labels: tuple[str, ...]
    needs_review: bool
    explicit_emergency: bool
    complete_input_seen: bool
    candidate_fingerprint: str | None
    raw: Mapping[str, Any] = field(repr=False, compare=False)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BoundaryDecision":
        if not isinstance(value, Mapping):
            raise ContractError("boundary decision must be a mapping")
        route = value.get("suggested_route", value.get("action"))
        if route not in VALID_BOUNDARY_ROUTES:
            raise ContractError(f"invalid boundary route: {route!r}")
        labels = _string_tuple(value.get("labels", ()), "boundary labels")
        if value.get("requires_rebuild_routing_adapter"):
            if value.get("schema_version") != "boundary-rebuild-1.0":
                raise ContractError("unexpected final boundary schema")
            if value.get("decision_kind") not in {"allow", "restrict", "uncertain", "unavailable"}:
                raise ContractError("unexpected final boundary decision")
            if set(_string_tuple(value.get("reasons", ()), "boundary reasons")) - {
                "diagnosis", "medication_decision", "clinical_treatment", "harmful_medical_instruction"}:
                raise ContractError("unknown final boundary reason")
            if set(_string_tuple(value.get("safety_signals", ()), "boundary safety signals")) - {
                "medical_emergency", "child_safeguarding", "suicide_safety"}:
                raise ContractError("unknown final boundary signal")
        screen = value.get("explicit_emergency_screen") or {}
        coverage = value.get("input_coverage") or {}
        return cls(
            suggested_route=route,
            status=str(value.get("status", value.get("model_status", "unknown"))),
            labels=labels,
            needs_review=bool(value.get("needs_review", False)),
            explicit_emergency=bool(screen.get("explicit_signal", False)) if isinstance(screen, Mapping) else False,
            complete_input_seen=bool(coverage.get("complete_input_seen", True)) if isinstance(coverage, Mapping) else True,
            candidate_fingerprint=(
                str(value["candidate_fingerprint"]) if value.get("candidate_fingerprint") else None
            ),
            raw=value,
        )

    @classmethod
    def fail_closed(cls, status: str = "component_error") -> "BoundaryDecision":
        return cls.from_mapping({
            "suggested_route": "safe_completion",
            "status": status,
            "labels": ["classifier_unavailable"],
            "needs_review": True,
            "input_coverage": {"complete_input_seen": False},
            "explicit_emergency_screen": {"explicit_signal": False},
        })


@dataclass(frozen=True)
class CombinedSafetyDecision:
    route: SystemRoute
    crisis: CrisisDecision
    boundary: BoundaryDecision
    degraded: bool
    degradation_reasons: tuple[str, ...]
    schema_version: str = field(default="combined-safety-decision/1.0", init=False)

    def audit_dict(self) -> dict[str, Any]:
        """Return privacy-minimal decision metadata; never raw message or spans."""
        return {
            "schema_version": self.schema_version,
            "route": self.route.value,
            "crisis_tier": self.crisis.tier,
            "crisis_categories": list(self.crisis.categories),
            "temporal_status": self.crisis.temporal_status,
            "severity": self.crisis.severity,
            "boundary_route": self.boundary.suggested_route,
            "boundary_status": self.boundary.status,
            "degraded": self.degraded,
            "degradation_reasons": list(self.degradation_reasons),
        }

    def to_policy_dict(self) -> dict[str, Any]:
        """Return the complete trusted input for the response-policy builder.

        This object is internal-only.  It may contain guardrail traces, but
        never the raw user message because neither upstream decision contains it.
        """
        result = {
            "schema_version": self.schema_version,
            "system_route": self.route.value,
            "crisis": dict(self.crisis.raw),
            "clinical_boundary": dict(self.boundary.raw),
            "degraded": self.degraded,
            "degradation_reasons": list(self.degradation_reasons),
        }
        if self.boundary.raw.get("requires_rebuild_routing_adapter"):
            from .boundary_policy import boundary_handling
            result["boundary_handling"] = boundary_handling(self.crisis, self.boundary)
        return result


@dataclass(frozen=True)
class Citation:
    source_id: str
    title: str
    source: str
    page: str | None = None
    url: str | None = None


@dataclass(frozen=True)
class PipelineResponse:
    request_id: str
    response: str
    route: str
    citations: tuple[Citation, ...] = ()
    resource_payload: Mapping[str, Any] | None = None
    used_fallback: bool = False
    output_guard_status: str = "unknown"
    schema_version: str = field(default="pipeline-response/1.0", init=False)

    def public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "response": self.response,
            "route": self.route,
            "citations": [asdict(item) for item in self.citations],
            "resources": dict(self.resource_payload) if self.resource_payload else None,
            "used_fallback": self.used_fallback,
        }


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ContractError(f"{field_name} must be a list of strings")
    return tuple(dict.fromkeys(value))
