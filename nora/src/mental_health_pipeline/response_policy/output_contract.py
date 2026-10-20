"""Independent output-guard contract and fail-closed resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .fallback_templates import FallbackResponse, render_fallback
from .schemas import ResponsePolicyPacket


OUTPUT_GUARD_SCHEMA_VERSION = "output-guard-decision/1.0"


class OutputGuardAction(str, Enum):
    APPROVE = "approve"
    RETRY = "retry"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class OutputGuardDecision:
    """Content-free verdict returned after inspecting a complete candidate."""

    action: OutputGuardAction
    guard_version: str
    policy_version: str
    output_guard_profile: str
    reason_codes: tuple[str, ...] = ()
    evaluated_complete_response: bool = True
    schema_version: str = OUTPUT_GUARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OUTPUT_GUARD_SCHEMA_VERSION:
            raise ValueError("Unsupported output-guard decision schema")
        if not self.guard_version or not self.policy_version or not self.output_guard_profile:
            raise ValueError("Output-guard identity and policy fields are required")
        if not self.evaluated_complete_response and self.action == OutputGuardAction.APPROVE:
            raise ValueError("A partial response cannot be approved")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("reason_codes must be de-duplicated")
        if self.action != OutputGuardAction.APPROVE and not self.reason_codes:
            raise ValueError("Non-approval decisions require a reason code")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action": self.action.value,
            "guard_version": self.guard_version,
            "policy_version": self.policy_version,
            "output_guard_profile": self.output_guard_profile,
            "reason_codes": list(self.reason_codes),
            "evaluated_complete_response": self.evaluated_complete_response,
        }


class OutputGuard(Protocol):
    """A real implementation must inspect the fully buffered candidate."""

    def inspect(
        self,
        candidate_response: str,
        policy: ResponsePolicyPacket,
    ) -> OutputGuardDecision: ...


class UnavailableOutputGuard:
    """Safe placeholder: it never authorizes generated text."""

    guard_version = "output-guard-unavailable/1.0"

    def inspect(
        self,
        candidate_response: str,
        policy: ResponsePolicyPacket,
    ) -> OutputGuardDecision:
        del candidate_response
        return OutputGuardDecision(
            action=OutputGuardAction.FALLBACK,
            guard_version=self.guard_version,
            policy_version=policy.policy_version,
            output_guard_profile=policy.output_guard_profile,
            reason_codes=("output_guard_unavailable",),
        )


@dataclass(frozen=True, slots=True)
class OutputResolution:
    """Result ready for the display layer; metadata remains content-free."""

    display_text: str | None
    approved_generated_response: bool
    retry_allowed: bool
    required_visible_elements: tuple[str, ...]
    fallback: FallbackResponse | None = None
    reason_codes: tuple[str, ...] = ()


def resolve_guarded_output(
    candidate_response: str,
    policy: ResponsePolicyPacket,
    decision: OutputGuardDecision,
    *,
    retry_count: int = 0,
    max_retries: int = 1,
) -> OutputResolution:
    """Resolve approval/retry/fallback without ever leaking a rejected candidate."""

    identity_matches = (
        decision.policy_version == policy.policy_version
        and decision.output_guard_profile == policy.output_guard_profile
    )
    candidate_valid = isinstance(candidate_response, str) and bool(candidate_response.strip())
    if (
        decision.action == OutputGuardAction.APPROVE
        and decision.evaluated_complete_response
        and identity_matches
        and candidate_valid
        and policy.generation_allowed
    ):
        return OutputResolution(
            display_text=candidate_response,
            approved_generated_response=True,
            retry_allowed=False,
            required_visible_elements=policy.required_visible_elements,
        )

    if (
        decision.action == OutputGuardAction.RETRY
        and identity_matches
        and policy.generation_allowed
        and retry_count < max_retries
    ):
        return OutputResolution(
            display_text=None,
            approved_generated_response=False,
            retry_allowed=True,
            required_visible_elements=policy.required_visible_elements,
            reason_codes=decision.reason_codes,
        )

    fallback = render_fallback(policy)
    reasons = list(decision.reason_codes)
    if not identity_matches:
        reasons.append("output_guard_policy_identity_mismatch")
    if not candidate_valid:
        reasons.append("empty_or_invalid_generated_response")
    if not policy.generation_allowed:
        reasons.append("generation_not_allowed_by_policy")
    return OutputResolution(
        display_text=fallback.text,
        approved_generated_response=False,
        retry_allowed=False,
        required_visible_elements=fallback.required_visible_elements,
        fallback=fallback,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )

