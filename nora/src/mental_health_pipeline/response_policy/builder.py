"""Build trusted response-policy packets from combined guardrail decisions."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import re
from typing import Any

from .compose import compose_policy
from .schemas import (
    BoundaryRoute,
    CrisisTier,
    PolicyBuildResult,
    PolicyStateUpdate,
    ResponsePolicyPacket,
    Severity,
    SystemRoute,
    TemporalStatus,
)
from .situation_codes import CRISIS_CATEGORIES, RuleSignal, derive_situation_codes


_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.:/-]{1,128}$")
_ROUTE_RANK = {
    SystemRoute.NORMAL_CHAT: 0,
    SystemRoute.MONITORED_SUPPORT: 1,
    SystemRoute.CLINICAL_BOUNDARY_SAFE_COMPLETION: 2,
    SystemRoute.UNSUPPORTED_SCOPE: 3,
    SystemRoute.URGENT_MEDICAL_RESPONSE: 4,
    SystemRoute.CRISIS_RESPONSE: 5,
}
_CATEGORY_ORDER = {name: index for index, name in enumerate(CRISIS_CATEGORIES)}
_EVIDENCE_ORDER = {
    "rules": 0,
    "classifier": 1,
    "classifier_near_threshold": 2,
    "suppressed_rules": 3,
    "upstream_decision": 4,
}


class PolicyBuildError(ValueError):
    """Raised only when the outer decision cannot be represented safely."""


class ResponsePolicyBuilder:
    """Translate upstream metadata without reclassifying the raw message.

    The builder accepts a mapping or an object exposing ``to_dict``.  It never
    accepts the raw user message and therefore cannot accidentally concatenate
    user-controlled text into trusted policy guidance.
    """

    def build(
        self,
        combined_decision: Mapping[str, Any] | Any,
        *,
        recent_state: Any | None = None,
    ) -> ResponsePolicyPacket:
        combined = _as_mapping(combined_decision, "combined_decision")
        flags: list[str] = []
        fatal_degraded = False

        crisis_raw = combined.get("crisis")
        if not isinstance(crisis_raw, Mapping):
            crisis: Mapping[str, Any] = {}
            flags.append("crisis_decision_missing_or_invalid")
            fatal_degraded = True
        else:
            crisis = crisis_raw

        boundary_raw = combined.get("clinical_boundary", combined.get("boundary"))
        if not isinstance(boundary_raw, Mapping):
            boundary: Mapping[str, Any] = {}
            flags.append("boundary_decision_missing_or_invalid")
            fatal_degraded = True
        else:
            boundary = boundary_raw

        tier = _enum_or_default(
            CrisisTier,
            crisis.get("tier"),
            CrisisTier.MONITORED if not crisis else CrisisTier.NORMAL,
            flags,
            "invalid_crisis_tier",
        )
        severity = _enum_or_default(
            Severity,
            crisis.get("severity"),
            Severity.WATCH if tier == CrisisTier.MONITORED else Severity.NONE,
            flags,
            "invalid_crisis_severity",
        )
        temporal = _enum_or_default(
            TemporalStatus,
            crisis.get("temporal_status"),
            TemporalStatus.UNSPECIFIED,
            flags,
            "invalid_crisis_temporal_status",
        )

        boundary_route_value = boundary.get(
            "suggested_route",
            boundary.get("action", boundary.get("route")),
        )
        boundary_route = _enum_or_default(
            BoundaryRoute,
            boundary_route_value,
            BoundaryRoute.SAFE_COMPLETION if not boundary else BoundaryRoute.ALLOW,
            flags,
            "invalid_boundary_route",
        )
        boundary_status = boundary.get("status", "ok")
        if not isinstance(boundary_status, str) or boundary_status != "ok":
            _append_unique(flags, "boundary_abstained_or_unavailable")
            if boundary_route == BoundaryRoute.ALLOW:
                boundary_route = BoundaryRoute.SAFE_COMPLETION
            fatal_degraded = True
        if bool(boundary.get("needs_review")) and boundary_route == BoundaryRoute.ALLOW:
            boundary_route = BoundaryRoute.SAFE_COMPLETION
            _append_unique(flags, "boundary_review_prevents_allow")
        if boundary.get("patient_facing_execution_authorized") is False:
            _append_unique(flags, "boundary_shadow_not_clinically_authorized")

        classifier = _classifier_mapping(crisis)
        if classifier.get("status", "ok") != "ok":
            _append_unique(flags, "crisis_abstained_or_unavailable")
            fatal_degraded = True
        classifier_labels = _clean_tokens(classifier.get("predicted_labels", ()), flags, "invalid_classifier_label")
        near_labels = _clean_tokens(classifier.get("near_threshold_labels", ()), flags, "invalid_near_threshold_label")
        rule_signals = _parse_rule_signals(crisis, flags)

        top_categories = _clean_tokens(crisis.get("categories", ()), flags, "invalid_crisis_label")
        primary = crisis.get("primary_category")
        if isinstance(primary, str) and primary != "no_crisis" and _is_safe_token(primary):
            top_categories.append(primary)
        elif primary not in {None, "no_crisis"}:
            _append_unique(flags, "invalid_primary_crisis_label")
        top_categories.extend(classifier_labels)
        top_categories.extend(near_labels)
        top_categories.extend(
            signal.category for signal in rule_signals if signal.category is not None
        )
        crisis_labels = tuple(sorted(
            set(top_categories) - {"no_crisis"},
            key=lambda item: (_CATEGORY_ORDER.get(item, len(_CATEGORY_ORDER)), item),
        ))
        unknown_crisis = sorted(set(crisis_labels) - set(CRISIS_CATEGORIES))
        if unknown_crisis:
            _append_unique(flags, "unknown_crisis_label_requires_review")

        topic_tags = tuple(sorted(set(_clean_tokens(
            crisis.get("topic_tags", ()), flags, "invalid_topic_tag"
        ))))
        boundary_labels = tuple(sorted(set(_clean_tokens(
            boundary.get("labels", ()), flags, "invalid_boundary_label"
        ))))
        boundary_review_reasons = tuple(sorted(set(_clean_tokens(
            boundary.get("review_reasons", ()), flags, "invalid_boundary_review_reason"
        ))))

        evidence_sources = _evidence_sources(
            crisis_labels, classifier_labels, near_labels, rule_signals
        )
        # Weak classifier-only monitoring stays generic so a false positive
        # cannot seed an invented abuse narrative. An ST01c crisis-specific
        # action selects conditional guidance grounded in explicit user words,
        # never an assumed timing, severity or intent that it does not assess.
        policy_labels = tuple(
            label
            for label in crisis_labels
            if set(evidence_sources.get(label, ()))
            & {"rules", "suppressed_rules", "upstream_decision"}
            or (tier == CrisisTier.CRISIS and label in CRISIS_CATEGORIES
                and isinstance(crisis.get("model_output"), Mapping)
                and crisis["model_output"].get("action") == "crisis_specific_route")
        )
        if policy_labels and isinstance(crisis.get("model_output"), Mapping):
            _append_unique(flags, "static_crisis_category_guidance_is_conditional")
        if len(policy_labels) != len(crisis_labels):
            _append_unique(flags, "classifier_only_crisis_signal_generalized")
        codes = derive_situation_codes(policy_labels, rule_signals, temporal, topic_tags)

        derived_route = _derive_route(tier, boundary_route)
        supplied_route = _parse_optional_route(combined.get("system_route"), flags)
        if supplied_route is not None and supplied_route != derived_route:
            _append_unique(flags, "upstream_route_mismatch_resolved_without_downgrade")
            system_route = max((supplied_route, derived_route), key=_ROUTE_RANK.__getitem__)
        else:
            system_route = supplied_route or derived_route

        recent_danger = bool(getattr(recent_state, "current_or_imminent_seen", False)) or (
            getattr(recent_state, "highest_recent_tier", None) == CrisisTier.CRISIS
            and bool(getattr(recent_state, "direct_safety_question_pending", False))
        )
        if recent_danger and _ROUTE_RANK[system_route] < _ROUTE_RANK[SystemRoute.MONITORED_SUPPORT]:
            system_route = SystemRoute.MONITORED_SUPPORT
            _append_unique(flags, "recent_danger_prevents_silent_normal_route")

        generation_allowed = (
            not fatal_degraded and system_route != SystemRoute.UNSUPPORTED_SCOPE
        )
        composition = compose_policy(
            codes,
            topic_tags,
            boundary_route,
            boundary_labels,
            system_route,
            recent_danger_context=recent_danger,
            generation_allowed=generation_allowed,
        )
        needs_clarification = (
            composition.needs_clarification
            or bool(unknown_crisis)
            or "boundary_review_prevents_allow" in flags
        )

        active_kinds = tuple(sorted({
            signal.kind for signal in rule_signals if not signal.suppressed
        }))
        suppressed_kinds = tuple(sorted({
            signal.kind for signal in rule_signals if signal.suppressed
        }))
        risk_signal_turn = bool(
            crisis_labels
            or topic_tags
            or tier != CrisisTier.NORMAL
            or system_route != SystemRoute.NORMAL_CHAT
        )
        direct_question = any(goal in {
            "ask_one_immediate_safety_question",
            "clarify_immediate_safety",
            "clarify_immediate_danger_to_others",
            "confirm_current_safety_after_prior_danger",
        } for goal in composition.response_goals)
        state_update = PolicyStateUpdate(
            categories=crisis_labels,
            risk_signal_turn=risk_signal_turn,
            highest_turn_tier=tier,
            highest_turn_severity=severity,
            temporal_status=temporal,
            direct_safety_question_requested=direct_question,
        )

        return ResponsePolicyPacket(
            system_route=system_route,
            situation_codes=tuple(code.value for code in codes),
            crisis_labels=crisis_labels,
            rule_kinds=active_kinds,
            suppressed_rule_kinds=suppressed_kinds,
            evidence_sources=evidence_sources,
            temporal_status=temporal,
            severity=severity,
            crisis_tier=tier,
            boundary_route=boundary_route,
            boundary_labels=boundary_labels,
            boundary_review_reasons=boundary_review_reasons,
            topic_tags=topic_tags,
            needs_clarification=needs_clarification,
            generator_guidance_id=composition.generator_guidance_id,
            secondary_guidance_ids=composition.secondary_guidance_ids,
            tone=composition.tone,
            response_goals=composition.response_goals,
            prohibited_behaviors=composition.prohibited_behaviors,
            required_visible_elements=composition.required_visible_elements,
            generation_allowed=composition.generation_allowed,
            fixed_content_timing=composition.fixed_content_timing,
            fallback_template_id=composition.fallback_template_id,
            output_guard_profile=composition.output_guard_profile,
            policy_flags=tuple(flags),
            state_update=state_update,
        )

    def build_with_state(
        self,
        combined_decision: Mapping[str, Any] | Any,
        *,
        recent_state: Any | None = None,
        turn_index: int | None = None,
        resources_shown: bool = False,
        safety_question_answered: bool = False,
    ) -> PolicyBuildResult:
        """Build a packet and atomically advance the content-free session state."""

        from .session_state import RecentSafetyState, update_recent_safety_state

        state = recent_state or RecentSafetyState()
        if not isinstance(state, RecentSafetyState):
            raise PolicyBuildError("recent_state must be a RecentSafetyState")
        if turn_index is None:
            turn_index = state.last_updated_turn + 1
        packet = self.build(combined_decision, recent_state=state)
        updated = update_recent_safety_state(
            state,
            packet.state_update,
            turn_index=turn_index,
            resources_shown=resources_shown,
            safety_question_answered=safety_question_answered,
        )
        return PolicyBuildResult(packet=packet, recent_state=updated)


def build_response_policy(
    combined_decision: Mapping[str, Any] | Any,
    *,
    recent_state: Any | None = None,
) -> ResponsePolicyPacket:
    """Convenience entry point for application orchestration."""

    return ResponsePolicyBuilder().build(combined_decision, recent_state=recent_state)


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return converted
    raise PolicyBuildError(f"{name} must be a mapping or expose to_dict()")


def _enum_or_default(enum_type, value, default, flags: list[str], flag: str):
    if isinstance(value, enum_type):
        return value
    if isinstance(value, Enum):
        value = value.value
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        _append_unique(flags, flag)
        return default


def _parse_optional_route(value: Any, flags: list[str]) -> SystemRoute | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    try:
        return SystemRoute(value)
    except (TypeError, ValueError):
        _append_unique(flags, "invalid_upstream_system_route")
        return None


def _derive_route(tier: CrisisTier, boundary_route: BoundaryRoute) -> SystemRoute:
    if tier == CrisisTier.CRISIS:
        return SystemRoute.CRISIS_RESPONSE
    if boundary_route == BoundaryRoute.URGENT_MEDICAL_ESCALATION:
        return SystemRoute.URGENT_MEDICAL_RESPONSE
    if boundary_route == BoundaryRoute.SAFE_COMPLETION:
        return SystemRoute.CLINICAL_BOUNDARY_SAFE_COMPLETION
    if tier == CrisisTier.MONITORED:
        return SystemRoute.MONITORED_SUPPORT
    return SystemRoute.NORMAL_CHAT


def _classifier_mapping(crisis: Mapping[str, Any]) -> Mapping[str, Any]:
    classifier = crisis.get("classifier")
    if isinstance(classifier, Mapping):
        return classifier
    trace = crisis.get("rule_trace")
    if isinstance(trace, Mapping):
        channel = trace.get("classifier_channel")
        if isinstance(channel, Mapping):
            return channel
    return {}


def _parse_rule_signals(crisis: Mapping[str, Any], flags: list[str]) -> tuple[RuleSignal, ...]:
    trace = crisis.get("rule_trace")
    channel = trace.get("rule_channel") if isinstance(trace, Mapping) else None
    raw_matches = channel.get("matches", ()) if isinstance(channel, Mapping) else ()
    if isinstance(raw_matches, (str, bytes)) or not isinstance(raw_matches, (list, tuple)):
        _append_unique(flags, "invalid_rule_trace")
        return ()
    signals: list[RuleSignal] = []
    for raw in raw_matches:
        if not isinstance(raw, Mapping):
            _append_unique(flags, "invalid_rule_trace_entry")
            continue
        kind = raw.get("kind")
        category = raw.get("category")
        if not isinstance(kind, str) or not _is_safe_token(kind):
            _append_unique(flags, "invalid_rule_kind")
            continue
        if category is not None and (not isinstance(category, str) or not _is_safe_token(category)):
            _append_unique(flags, "invalid_rule_category")
            category = None
        temporal = _enum_or_default(
            TemporalStatus,
            raw.get("temporal"),
            TemporalStatus.UNSPECIFIED,
            flags,
            "invalid_rule_temporal_status",
        )
        reason = raw.get("suppression_reason")
        if reason is not None and (not isinstance(reason, str) or not _is_safe_token(reason)):
            reason = "invalid_or_redacted_reason"
            _append_unique(flags, "invalid_rule_suppression_reason")
        signals.append(RuleSignal(
            category=category,
            kind=kind,
            temporal=temporal,
            suppressed=bool(raw.get("suppressed", False)),
            suppression_reason=reason,
        ))
    return tuple(signals)


def _evidence_sources(
    labels: tuple[str, ...],
    classifier_labels: list[str],
    near_labels: list[str],
    signals: tuple[RuleSignal, ...],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    classifier_set = set(classifier_labels)
    near_set = set(near_labels)
    for label in labels:
        sources: set[str] = set()
        if any(signal.category == label and not signal.suppressed for signal in signals):
            sources.add("rules")
        if label in classifier_set:
            sources.add("classifier")
        if label in near_set:
            sources.add("classifier_near_threshold")
        if any(signal.category == label and signal.suppressed for signal in signals):
            sources.add("suppressed_rules")
        if not sources:
            sources.add("upstream_decision")
        result[label] = tuple(sorted(sources, key=_EVIDENCE_ORDER.__getitem__))
    return result


def _clean_tokens(value: Any, flags: list[str], flag: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set, frozenset)):
        _append_unique(flags, flag)
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, Enum):
            item = item.value
        if isinstance(item, str) and _is_safe_token(item):
            result.append(item)
        else:
            _append_unique(flags, flag)
    return result


def _is_safe_token(value: str) -> bool:
    return bool(_SAFE_TOKEN.fullmatch(value))


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)

