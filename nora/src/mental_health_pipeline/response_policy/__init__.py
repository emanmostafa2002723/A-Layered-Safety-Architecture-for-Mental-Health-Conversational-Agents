"""Response-policy translation for the mental-health pipeline.

This package converts trusted crisis/boundary decisions into generation
constraints.  It does not classify user text and does not constitute clinical
validation or patient-facing authorization.
"""

from .builder import PolicyBuildError, ResponsePolicyBuilder, build_response_policy
from .fallback_templates import (
    FallbackResponse,
    FallbackTemplate,
    get_fallback_template,
    render_fallback,
)
from .output_contract import (
    OutputGuard,
    OutputGuardAction,
    OutputGuardDecision,
    OutputResolution,
    UnavailableOutputGuard,
    resolve_guarded_output,
)
from .render import build_generation_messages, render_response_policy
from .schemas import (
    BoundaryRoute,
    CrisisTier,
    POLICY_SCHEMA_VERSION,
    POLICY_VERSION,
    PolicyBuildResult,
    PolicyStateUpdate,
    ResponsePolicyPacket,
    Severity,
    SystemRoute,
    TemporalStatus,
)
from .session_state import (
    MAX_AGE_TURNS,
    MAX_OBSERVATIONS,
    RecentSafetyState,
    SafetyObservation,
    should_repeat_resources,
    update_recent_safety_state,
)
from .situation_codes import RuleSignal, SituationCode, derive_situation_codes

__all__ = [
    "BoundaryRoute",
    "CrisisTier",
    "FallbackResponse",
    "FallbackTemplate",
    "MAX_AGE_TURNS",
    "MAX_OBSERVATIONS",
    "OutputGuard",
    "OutputGuardAction",
    "OutputGuardDecision",
    "OutputResolution",
    "POLICY_SCHEMA_VERSION",
    "POLICY_VERSION",
    "PolicyBuildError",
    "PolicyBuildResult",
    "PolicyStateUpdate",
    "RecentSafetyState",
    "ResponsePolicyBuilder",
    "ResponsePolicyPacket",
    "RuleSignal",
    "SafetyObservation",
    "Severity",
    "SituationCode",
    "SystemRoute",
    "TemporalStatus",
    "UnavailableOutputGuard",
    "build_generation_messages",
    "build_response_policy",
    "derive_situation_codes",
    "get_fallback_template",
    "render_fallback",
    "render_response_policy",
    "resolve_guarded_output",
    "should_repeat_resources",
    "update_recent_safety_state",
]

