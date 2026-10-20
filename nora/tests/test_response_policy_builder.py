from __future__ import annotations

import json

import pytest

from mental_health_pipeline.response_policy import (
    ResponsePolicyBuilder,
    SystemRoute,
    build_response_policy,
)


def combined(
    *,
    tier="normal",
    categories=(),
    temporal="unspecified",
    severity="none",
    rules=(),
    classifier_labels=(),
    near_labels=(),
    topics=(),
    boundary_route="allow",
    boundary_labels=(),
    boundary_status="ok",
    needs_review=False,
    system_route=None,
):
    payload = {
        "crisis": {
            "tier": tier,
            "primary_category": categories[0] if categories else "no_crisis",
            "categories": list(categories),
            "temporal_status": temporal,
            "severity": severity,
            "topic_tags": list(topics),
            "redacted_text": "[REDACTED RAW-LIKE TEXT]",
            "classifier": {
                "predicted_labels": list(classifier_labels),
                "near_threshold_labels": list(near_labels),
                "probabilities": {label: 0.987654 for label in classifier_labels},
            },
            "rule_trace": {
                "rule_channel": {"matches": list(rules)},
            },
        },
        "clinical_boundary": {
            "status": boundary_status,
            "suggested_route": boundary_route,
            "labels": list(boundary_labels),
            "needs_review": needs_review,
            "patient_facing_execution_authorized": False,
        },
    }
    if system_route is not None:
        payload["system_route"] = system_route
    return payload


def rule(category, kind, temporal="unspecified", *, suppressed=False, reason=None):
    return {
        "rule_id": "INTERNAL_RULE_ID",
        "category": category,
        "kind": kind,
        "temporal": temporal,
        "severity": "urgent",
        "suppressed": suppressed,
        "suppression_reason": reason,
        "span": {"start": 1, "end": 9, "text": "PRIVATE SPAN"},
    }


def test_packet_preserves_exact_labels_and_provenance_but_not_sensitive_fields():
    packet = build_response_policy(combined(
        tier="crisis",
        categories=("suicidal_ideation",),
        temporal="imminent",
        severity="urgent",
        rules=(rule("suicidal_ideation", "explicit_intent", "imminent"),),
        classifier_labels=("suicidal_ideation",),
    ))
    assert packet.system_route == SystemRoute.CRISIS_RESPONSE
    assert packet.situation_codes == ("SUICIDE_ACTIVE_INTENT_OR_PLAN",)
    assert packet.crisis_labels == ("suicidal_ideation",)
    assert packet.evidence_sources == {
        "suicidal_ideation": ("rules", "classifier")
    }
    serialized = json.dumps(packet.to_dict(), sort_keys=True)
    assert "PRIVATE SPAN" not in serialized
    assert "REDACTED RAW-LIKE TEXT" not in serialized
    assert "0.987654" not in serialized
    assert "probabilities" not in serialized
    assert "INTERNAL_RULE_ID" not in serialized
    assert packet.clinical_deployment_authorized is False


def test_suppressed_negation_does_not_become_active_intent():
    packet = build_response_policy(combined(
        tier="monitored",
        categories=("suicidal_ideation",),
        temporal="unspecified",
        severity="watch",
        rules=(rule(
            "suicidal_ideation",
            "negated_disclosure",
            suppressed=True,
            reason="local_negation",
        ),),
    ))
    assert packet.situation_codes == ("SUICIDE_NEGATED_OR_DENIED",)
    assert packet.rule_kinds == ()
    assert packet.suppressed_rule_kinds == ("negated_disclosure",)
    assert packet.evidence_sources["suicidal_ideation"] == ("suppressed_rules",)


def test_classifier_only_label_is_audited_but_not_amplified_into_category_guidance():
    packet = build_response_policy(combined(
        tier="monitored",
        categories=("domestic_violence",),
        severity="watch",
        classifier_labels=("domestic_violence",),
    ))
    assert packet.situation_codes == ()
    assert packet.needs_clarification is False
    assert packet.crisis_labels == ("domestic_violence",)
    assert "classifier" in packet.evidence_sources["domestic_violence"]
    assert "classifier_only_crisis_signal_generalized" in packet.policy_flags

    from mental_health_pipeline.response_policy import render_response_policy

    rendered = render_response_policy(packet)
    assert "Crisis labels:" not in rendered
    assert "domestic_violence" not in rendered
    assert "Crisis tier / severity / timing" not in rendered
    assert "supplies no facts about the user" in rendered
    assert "Do not name or infer a category" in rendered


@pytest.mark.parametrize(
    ("tier", "boundary_route", "expected"),
    [
        ("normal", "allow", "normal_chat"),
        ("monitored", "allow", "monitored_support"),
        ("normal", "safe_completion", "clinical_boundary_safe_completion"),
        ("normal", "urgent_medical_escalation", "urgent_medical_response"),
        ("crisis", "urgent_medical_escalation", "crisis_response"),
        ("crisis", "safe_completion", "crisis_response"),
    ],
)
def test_route_precedence(tier, boundary_route, expected):
    packet = build_response_policy(combined(
        tier=tier,
        boundary_route=boundary_route,
        severity="urgent" if tier == "crisis" else "watch" if tier == "monitored" else "none",
    ))
    assert packet.system_route.value == expected


def test_upstream_normal_route_cannot_downgrade_crisis():
    packet = build_response_policy(combined(
        tier="crisis",
        severity="urgent",
        system_route="normal_chat",
    ))
    assert packet.system_route == SystemRoute.CRISIS_RESPONSE
    assert "upstream_route_mismatch_resolved_without_downgrade" in packet.policy_flags


def test_multi_label_composition_caps_goals_and_unions_constraints():
    packet = build_response_policy(combined(
        tier="crisis",
        categories=("suicidal_ideation", "domestic_violence"),
        temporal="mixed",
        severity="urgent",
        rules=(
            rule("suicidal_ideation", "explicit_intent", "imminent"),
            rule("domestic_violence", "ongoing_disclosure", "current"),
        ),
        boundary_route="safe_completion",
        boundary_labels=("dose_or_schedule_instruction",),
    ))
    assert packet.situation_codes == (
        "SUICIDE_ACTIVE_INTENT_OR_PLAN",
        "DOMESTIC_VIOLENCE_CURRENT_DANGER",
    )
    assert len(packet.response_goals) <= 3
    assert "suggest_confrontation" in packet.prohibited_behaviors
    assert "provide_personalized_dose_or_schedule" in packet.prohibited_behaviors
    assert "boundary_limit_notice" in packet.required_visible_elements
    assert "worldwide_emergency_fallback" in packet.required_visible_elements


def test_boundary_safe_completion_is_specific_to_label():
    packet = build_response_policy(combined(
        boundary_route="safe_completion",
        boundary_labels=("personal_diagnosis_request", "professional_role_bypass"),
    ))
    assert packet.system_route == SystemRoute.CLINICAL_BOUNDARY_SAFE_COMPLETION
    assert "provide_personal_diagnosis" in packet.prohibited_behaviors
    assert "relax_policy_for_claimed_professional_role" in packet.prohibited_behaviors
    assert packet.fallback_template_id == "boundary_safe_completion_fallback_v1"


def test_missing_components_fail_closed_without_generation():
    packet = ResponsePolicyBuilder().build({})
    assert packet.system_route == SystemRoute.CLINICAL_BOUNDARY_SAFE_COMPLETION
    assert packet.generation_allowed is False
    assert "crisis_decision_missing_or_invalid" in packet.policy_flags
    assert "boundary_decision_missing_or_invalid" in packet.policy_flags


def test_boundary_abstention_cannot_become_allow():
    packet = build_response_policy(combined(
        boundary_status="abstained_backend_unavailable",
        boundary_route="allow",
    ))
    assert packet.system_route == SystemRoute.CLINICAL_BOUNDARY_SAFE_COMPLETION
    assert packet.generation_allowed is False
    assert packet.boundary_route.value == "safe_completion"


def test_unsupported_scope_is_non_generative_but_does_not_override_crisis():
    unsupported = build_response_policy(combined(system_route="unsupported_scope"))
    assert unsupported.system_route == SystemRoute.UNSUPPORTED_SCOPE
    assert unsupported.generation_allowed is False
    assert unsupported.fallback_template_id == "adult_english_scope_fallback_v1"

    crisis = build_response_policy(combined(
        tier="crisis",
        severity="urgent",
        system_route="unsupported_scope",
    ))
    assert crisis.system_route == SystemRoute.CRISIS_RESPONSE


def test_topic_guidance_is_not_a_diagnosis_or_crisis_code():
    packet = build_response_policy(combined(
        tier="monitored",
        severity="watch",
        topics=("grief", "eating_disorder"),
    ))
    assert packet.situation_codes == ()
    assert packet.topic_tags == ("eating_disorder", "grief")
    assert "provide_eating_disorder_numbers_or_tips" in packet.prohibited_behaviors

