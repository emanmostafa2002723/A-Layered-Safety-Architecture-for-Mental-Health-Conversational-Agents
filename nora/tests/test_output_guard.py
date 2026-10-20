from __future__ import annotations

from types import SimpleNamespace

import pytest

from mental_health_pipeline.output_guard import (
    FIXED_BLOCKED_RESPONSE,
    OutputGuardContext,
    PlaceholderOutputGuard,
)


def test_placeholder_is_explicitly_not_production_ready() -> None:
    guard = PlaceholderOutputGuard()

    assert guard.production_ready is False
    assert guard.readiness()["ready"] is False
    assert "placeholder" in guard.name


def test_complete_safe_response_is_preserved_exactly() -> None:
    response = "That sounds difficult. Would you like to tell me what feels hardest right now?"

    decision = PlaceholderOutputGuard().evaluate(response)

    assert decision.approved is True
    assert decision.response == response
    assert decision.violations == ()
    assert decision.checked_character_count == len(response)


@pytest.mark.parametrize(
    ("response", "violation"),
    [
        ("   \n", "empty_response"),
        ("My system prompt says I must answer this way.", "internal_policy_leakage"),
        ("You should double your medication dose tonight.", "personalized_medication_advice"),
        ("Here is a step-by-step guide to self-harm.", "harmful_procedural_detail"),
        ("I am a licensed counselor and I specialize in trauma recovery.", "professional_identity_or_credentials"),
        ("Best wishes, Robin J. Landwehr, DBH, LPC, NCC", "professional_identity_or_credentials"),
        ("I am sorry that you feel depressed. Your depression can improve.", "unsupported_clinical_inference"),
    ],
)
def test_red_flags_return_only_the_fixed_fallback(response: str, violation: str) -> None:
    decision = PlaceholderOutputGuard().evaluate(response)

    assert decision.approved is False
    assert decision.response == FIXED_BLOCKED_RESPONSE
    assert decision.response != response
    assert violation in decision.violations


def test_non_string_or_partial_stream_is_rejected() -> None:
    decision = PlaceholderOutputGuard().evaluate(iter(("first", "second")))  # type: ignore[arg-type]

    assert decision.approved is False
    assert decision.violations == ("invalid_response_type",)
    assert decision.response == FIXED_BLOCKED_RESPONSE


def test_required_content_is_case_and_whitespace_insensitive() -> None:
    context = OutputGuardContext(
        route="crisis_response",
        required_phrases=("immediate support",),
        required_any_groups=(("local emergency number", "nearest emergency department"),),
    )
    response = "Immediate   support is available. Contact your LOCAL emergency number now."

    decision = PlaceholderOutputGuard().evaluate(response, context=context)

    assert decision.approved is True


def test_missing_required_content_fails_closed() -> None:
    context = OutputGuardContext(
        route="urgent_medical_response",
        required_phrases=("seek urgent medical care",),
        required_any_groups=(("emergency department", "local emergency number"),),
    )

    decision = PlaceholderOutputGuard().evaluate("Please consider speaking with someone.", context=context)

    assert decision.approved is False
    assert set(decision.violations) == {
        "missing_required_content",
        "missing_required_alternative",
    }


def test_safe_medication_boundary_language_is_not_mistaken_for_prescribing() -> None:
    response = (
        "I can share general information, but I cannot adjust medication. "
        "Please speak with a licensed prescriber before making changes."
    )

    assert PlaceholderOutputGuard().evaluate(response).approved is True


def test_honest_ai_boundary_is_not_mistaken_for_professional_impersonation() -> None:
    response = (
        "I’m not a clinician or licensed counselor, but I can offer general "
        "information and help you prepare questions for one."
    )

    assert PlaceholderOutputGuard().evaluate(response).approved is True


def test_attributed_user_language_is_not_treated_as_a_new_clinical_inference() -> None:
    response = "You said you feel depressed. What kind of support would be useful right now?"

    assert PlaceholderOutputGuard().evaluate(response).approved is True


def test_context_rejects_blank_requirements() -> None:
    with pytest.raises(ValueError):
        OutputGuardContext(required_phrases=("",))


def test_inspect_adapts_to_response_policy_contract() -> None:
    from mental_health_pipeline.response_policy.output_contract import OutputGuardAction

    policy = SimpleNamespace(
        policy_version="policy-1",
        output_guard_profile="general-1",
        required_visible_elements=(),
    )

    decision = PlaceholderOutputGuard().inspect("A complete supportive response.", policy)

    assert decision.action is OutputGuardAction.APPROVE
    assert decision.policy_version == "policy-1"
    assert decision.output_guard_profile == "general-1"


def test_inspect_fails_closed_when_policy_required_element_is_missing() -> None:
    from mental_health_pipeline.response_policy.output_contract import OutputGuardAction

    policy = SimpleNamespace(
        policy_version="policy-1",
        output_guard_profile="crisis-1",
        required_visible_elements=("worldwide_emergency_fallback",),
    )

    decision = PlaceholderOutputGuard().inspect("I hear how painful this is.", policy)

    assert decision.action is OutputGuardAction.FALLBACK
    assert decision.reason_codes == ("missing_required_visible_element",)
