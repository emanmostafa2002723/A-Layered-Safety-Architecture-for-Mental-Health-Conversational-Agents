from __future__ import annotations

import pytest

from mental_health_pipeline.response_policy import (
    RuleSignal,
    SituationCode,
    TemporalStatus,
    derive_situation_codes,
)


@pytest.mark.parametrize(
    ("label", "kind", "temporal", "suppressed", "reason", "topic_tags", "expected"),
    [
        ("suicidal_ideation", "in_progress", "current", False, None, (), "SUICIDE_ATTEMPT_OR_IN_PROGRESS"),
        ("suicidal_ideation", "explicit_intent", "imminent", False, None, (), "SUICIDE_ACTIVE_INTENT_OR_PLAN"),
        ("suicidal_ideation", "preparation", "unspecified", False, None, (), "SUICIDE_PREPARATORY_SIGNALS"),
        ("suicidal_ideation", "ideation_disclosure", "current", False, None, (), "SUICIDE_ACTIVE_IDEATION_UNSPECIFIED_INTENT"),
        ("suicidal_ideation", "passive_ideation", "unspecified", False, None, (), "SUICIDE_PASSIVE_DEATH_WISH"),
        ("suicidal_ideation", "negated_disclosure", "unspecified", True, "local_negation", (), "SUICIDE_NEGATED_OR_DENIED"),
        ("suicidal_ideation", "ideation_disclosure", "past", False, None, (), "SUICIDE_HISTORICAL"),
        ("suicidal_ideation", None, "unspecified", False, None, (), "SUICIDE_SIGNAL_UNSPECIFIED"),
        ("self_directed_harm", "in_progress", "current", False, None, (), "SELF_HARM_IN_PROGRESS"),
        ("self_directed_harm", "direct_disclosure", "recent", False, None, (), "SELF_HARM_RECENT_DISCLOSURE"),
        ("self_directed_harm", "contextual_abbreviation", "unspecified", False, None, (), "SELF_HARM_URGE_OR_RELAPSE"),
        ("self_directed_harm", "negated_disclosure", "unspecified", True, "local_negation", (), "SELF_HARM_NEGATED_OR_DENIED"),
        ("self_directed_harm", "direct_disclosure", "past", False, None, (), "SELF_HARM_HISTORICAL"),
        ("self_directed_harm", None, "unspecified", False, None, (), "SELF_HARM_SIGNAL_UNSPECIFIED"),
        ("sexual_violence", "ongoing_disclosure", "current", False, None, (), "SEXUAL_VIOLENCE_CURRENT_DANGER"),
        ("sexual_violence", "direct_disclosure", "recent", False, None, (), "SEXUAL_VIOLENCE_RECENT"),
        ("sexual_violence", "direct_disclosure", "past", False, None, (), "SEXUAL_VIOLENCE_HISTORICAL_TRAUMA"),
        ("sexual_violence", None, "mixed", False, None, (), "SEXUAL_VIOLENCE_SIGNAL_UNSPECIFIED"),
        ("domestic_violence", "ongoing_disclosure", "current", False, None, (), "DOMESTIC_VIOLENCE_CURRENT_DANGER"),
        ("domestic_violence", "direct_disclosure", "recent", False, None, (), "DOMESTIC_VIOLENCE_RECENT"),
        ("domestic_violence", "direct_disclosure", "past", False, None, (), "DOMESTIC_VIOLENCE_HISTORICAL"),
        ("domestic_violence", None, "mixed", False, None, (), "DOMESTIC_VIOLENCE_SIGNAL_UNSPECIFIED"),
        ("child_abuse_or_endangerment", "ongoing_disclosure", "current", False, None, (), "CHILD_ABUSE_CURRENT_DANGER"),
        ("child_abuse_or_endangerment", "third_party_endangerment", "current", False, None, (), "CHILD_ENDANGERMENT_THIRD_PARTY"),
        ("child_abuse_or_endangerment", "historical_disclosure", "past", False, None, (), "CHILDHOOD_ABUSE_HISTORICAL_TRAUMA"),
        ("child_abuse_or_endangerment", None, "unspecified", False, None, (), "CHILD_ABUSE_SIGNAL_UNSPECIFIED"),
        ("acute_crisis", "possible_medical_emergency", "current", False, None, (), "POSSIBLE_OVERDOSE_OR_POISONING"),
        ("acute_crisis", "acute_distress", "current", False, None, (), "PANIC_OR_ACUTE_DISTRESS_CURRENT"),
        ("acute_crisis", None, "unspecified", False, None, ("substance_use",), "SUBSTANCE_WITHDRAWAL_OR_ACUTE_SUBSTANCE_CONCERN"),
        ("acute_crisis", None, "unspecified", False, None, (), "ACUTE_CRISIS_SIGNAL_UNSPECIFIED"),
        ("harm_to_others", "explicit_intent", "imminent", False, None, (), "HARM_OTHERS_ACTIVE_INTENT"),
        ("harm_to_others", None, "past", False, None, (), "HARM_OTHERS_SIGNAL_UNSPECIFIED"),
    ],
)
def test_every_situation_code_is_derived_deterministically(
    label, kind, temporal, suppressed, reason, topic_tags, expected
):
    signals = () if kind is None else (
        RuleSignal(
            category=label,
            kind=kind,
            temporal=TemporalStatus(temporal),
            suppressed=suppressed,
            suppression_reason=reason,
        ),
    )
    codes = derive_situation_codes(
        (label,), signals, TemporalStatus(temporal), topic_tags
    )
    assert tuple(code.value for code in codes) == (expected,)


def test_possible_overdose_plus_suicide_is_attempt_and_overdose_with_attempt_first():
    signals = (
        RuleSignal(
            category="acute_crisis",
            kind="possible_medical_emergency",
            temporal=TemporalStatus.CURRENT,
        ),
    )
    codes = derive_situation_codes(
        ("suicidal_ideation", "acute_crisis"),
        signals,
        TemporalStatus.CURRENT,
    )
    assert codes == (
        SituationCode.SUICIDE_ATTEMPT_OR_IN_PROGRESS,
        SituationCode.POSSIBLE_OVERDOSE_OR_POISONING,
    )


def test_multi_label_priority_puts_current_danger_before_historical_trauma():
    signals = (
        RuleSignal("self_directed_harm", "in_progress", TemporalStatus.CURRENT),
        RuleSignal("sexual_violence", "historical_disclosure", TemporalStatus.PAST),
    )
    codes = derive_situation_codes(
        ("self_directed_harm", "sexual_violence"), signals, TemporalStatus.MIXED
    )
    assert codes[0] == SituationCode.SELF_HARM_IN_PROGRESS
    assert codes[1] == SituationCode.SEXUAL_VIOLENCE_HISTORICAL_TRAUMA


def test_rule_only_harm_to_others_is_not_lost():
    codes = derive_situation_codes(
        (),
        (RuleSignal("harm_to_others", "explicit_intent", TemporalStatus.IMMINENT),),
        TemporalStatus.IMMINENT,
    )
    assert codes == (SituationCode.HARM_OTHERS_ACTIVE_INTENT,)

