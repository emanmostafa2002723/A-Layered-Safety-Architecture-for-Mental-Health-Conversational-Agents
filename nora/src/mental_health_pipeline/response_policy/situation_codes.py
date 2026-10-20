"""Deterministic situation-code derivation from trusted guardrail metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .schemas import TemporalStatus


class SituationCode(str, Enum):
    SUICIDE_ATTEMPT_OR_IN_PROGRESS = "SUICIDE_ATTEMPT_OR_IN_PROGRESS"
    SUICIDE_ACTIVE_INTENT_OR_PLAN = "SUICIDE_ACTIVE_INTENT_OR_PLAN"
    SUICIDE_PREPARATORY_SIGNALS = "SUICIDE_PREPARATORY_SIGNALS"
    SUICIDE_ACTIVE_IDEATION_UNSPECIFIED_INTENT = "SUICIDE_ACTIVE_IDEATION_UNSPECIFIED_INTENT"
    SUICIDE_PASSIVE_DEATH_WISH = "SUICIDE_PASSIVE_DEATH_WISH"
    SUICIDE_NEGATED_OR_DENIED = "SUICIDE_NEGATED_OR_DENIED"
    SUICIDE_HISTORICAL = "SUICIDE_HISTORICAL"
    SUICIDE_SIGNAL_UNSPECIFIED = "SUICIDE_SIGNAL_UNSPECIFIED"
    SELF_HARM_IN_PROGRESS = "SELF_HARM_IN_PROGRESS"
    SELF_HARM_RECENT_DISCLOSURE = "SELF_HARM_RECENT_DISCLOSURE"
    SELF_HARM_URGE_OR_RELAPSE = "SELF_HARM_URGE_OR_RELAPSE"
    SELF_HARM_NEGATED_OR_DENIED = "SELF_HARM_NEGATED_OR_DENIED"
    SELF_HARM_HISTORICAL = "SELF_HARM_HISTORICAL"
    SELF_HARM_SIGNAL_UNSPECIFIED = "SELF_HARM_SIGNAL_UNSPECIFIED"
    SEXUAL_VIOLENCE_CURRENT_DANGER = "SEXUAL_VIOLENCE_CURRENT_DANGER"
    SEXUAL_VIOLENCE_RECENT = "SEXUAL_VIOLENCE_RECENT"
    SEXUAL_VIOLENCE_HISTORICAL_TRAUMA = "SEXUAL_VIOLENCE_HISTORICAL_TRAUMA"
    SEXUAL_VIOLENCE_SIGNAL_UNSPECIFIED = "SEXUAL_VIOLENCE_SIGNAL_UNSPECIFIED"
    DOMESTIC_VIOLENCE_CURRENT_DANGER = "DOMESTIC_VIOLENCE_CURRENT_DANGER"
    DOMESTIC_VIOLENCE_RECENT = "DOMESTIC_VIOLENCE_RECENT"
    DOMESTIC_VIOLENCE_HISTORICAL = "DOMESTIC_VIOLENCE_HISTORICAL"
    DOMESTIC_VIOLENCE_SIGNAL_UNSPECIFIED = "DOMESTIC_VIOLENCE_SIGNAL_UNSPECIFIED"
    CHILD_ABUSE_CURRENT_DANGER = "CHILD_ABUSE_CURRENT_DANGER"
    CHILD_ENDANGERMENT_THIRD_PARTY = "CHILD_ENDANGERMENT_THIRD_PARTY"
    CHILDHOOD_ABUSE_HISTORICAL_TRAUMA = "CHILDHOOD_ABUSE_HISTORICAL_TRAUMA"
    CHILD_ABUSE_SIGNAL_UNSPECIFIED = "CHILD_ABUSE_SIGNAL_UNSPECIFIED"
    POSSIBLE_OVERDOSE_OR_POISONING = "POSSIBLE_OVERDOSE_OR_POISONING"
    PANIC_OR_ACUTE_DISTRESS_CURRENT = "PANIC_OR_ACUTE_DISTRESS_CURRENT"
    SUBSTANCE_WITHDRAWAL_OR_ACUTE_SUBSTANCE_CONCERN = "SUBSTANCE_WITHDRAWAL_OR_ACUTE_SUBSTANCE_CONCERN"
    ACUTE_CRISIS_SIGNAL_UNSPECIFIED = "ACUTE_CRISIS_SIGNAL_UNSPECIFIED"
    HARM_OTHERS_ACTIVE_INTENT = "HARM_OTHERS_ACTIVE_INTENT"
    HARM_OTHERS_SIGNAL_UNSPECIFIED = "HARM_OTHERS_SIGNAL_UNSPECIFIED"


CRISIS_CATEGORIES = (
    "suicidal_ideation",
    "self_directed_harm",
    "harm_to_others",
    "child_abuse_or_endangerment",
    "sexual_violence",
    "domestic_violence",
    "acute_crisis",
)


@dataclass(frozen=True, slots=True)
class RuleSignal:
    category: str | None
    kind: str
    temporal: TemporalStatus = TemporalStatus.UNSPECIFIED
    suppressed: bool = False
    suppression_reason: str | None = None


# Lower values are more urgent.  Sorting by this table makes multi-label policy
# composition deterministic and prevents a historical signal from leading an
# urgent one.
SITUATION_PRIORITY: Mapping[SituationCode, int] = {
    SituationCode.SUICIDE_ATTEMPT_OR_IN_PROGRESS: 10,
    SituationCode.POSSIBLE_OVERDOSE_OR_POISONING: 11,
    SituationCode.SUICIDE_ACTIVE_INTENT_OR_PLAN: 20,
    SituationCode.HARM_OTHERS_ACTIVE_INTENT: 21,
    SituationCode.CHILD_ABUSE_CURRENT_DANGER: 30,
    SituationCode.SEXUAL_VIOLENCE_CURRENT_DANGER: 31,
    SituationCode.DOMESTIC_VIOLENCE_CURRENT_DANGER: 32,
    SituationCode.SELF_HARM_IN_PROGRESS: 40,
    SituationCode.SUICIDE_PREPARATORY_SIGNALS: 50,
    SituationCode.SUICIDE_ACTIVE_IDEATION_UNSPECIFIED_INTENT: 51,
    SituationCode.SUICIDE_PASSIVE_DEATH_WISH: 52,
    SituationCode.CHILD_ENDANGERMENT_THIRD_PARTY: 60,
    SituationCode.SELF_HARM_RECENT_DISCLOSURE: 61,
    SituationCode.SELF_HARM_URGE_OR_RELAPSE: 62,
    SituationCode.SEXUAL_VIOLENCE_RECENT: 63,
    SituationCode.DOMESTIC_VIOLENCE_RECENT: 64,
    SituationCode.PANIC_OR_ACUTE_DISTRESS_CURRENT: 65,
    SituationCode.SUBSTANCE_WITHDRAWAL_OR_ACUTE_SUBSTANCE_CONCERN: 66,
    SituationCode.SUICIDE_HISTORICAL: 70,
    SituationCode.SELF_HARM_HISTORICAL: 71,
    SituationCode.SEXUAL_VIOLENCE_HISTORICAL_TRAUMA: 72,
    SituationCode.DOMESTIC_VIOLENCE_HISTORICAL: 73,
    SituationCode.CHILDHOOD_ABUSE_HISTORICAL_TRAUMA: 74,
    SituationCode.SUICIDE_NEGATED_OR_DENIED: 80,
    SituationCode.SELF_HARM_NEGATED_OR_DENIED: 81,
    SituationCode.SUICIDE_SIGNAL_UNSPECIFIED: 90,
    SituationCode.SELF_HARM_SIGNAL_UNSPECIFIED: 91,
    SituationCode.SEXUAL_VIOLENCE_SIGNAL_UNSPECIFIED: 92,
    SituationCode.DOMESTIC_VIOLENCE_SIGNAL_UNSPECIFIED: 93,
    SituationCode.CHILD_ABUSE_SIGNAL_UNSPECIFIED: 94,
    SituationCode.ACUTE_CRISIS_SIGNAL_UNSPECIFIED: 95,
    SituationCode.HARM_OTHERS_SIGNAL_UNSPECIFIED: 96,
}


def derive_situation_codes(
    crisis_labels: Iterable[str],
    rule_signals: Sequence[RuleSignal],
    decision_temporal: TemporalStatus,
    topic_tags: Iterable[str] = (),
) -> tuple[SituationCode, ...]:
    """Return at most one code per category, ordered by safety priority."""

    labels = set(crisis_labels)
    signals_by_category = {
        category: [signal for signal in rule_signals if signal.category == category]
        for category in CRISIS_CATEGORIES
    }
    # Rule-only categories must still produce policy even if an adapter omitted
    # them from the top-level classifier categories.
    labels.update(
        signal.category for signal in rule_signals
        if signal.category in CRISIS_CATEGORIES
    )
    codes: list[SituationCode] = []
    for category in CRISIS_CATEGORIES:
        if category not in labels:
            continue
        signals = signals_by_category[category]
        if category == "suicidal_ideation":
            code = _suicide(signals, rule_signals, decision_temporal)
        elif category == "self_directed_harm":
            code = _self_harm(signals, decision_temporal)
        elif category == "sexual_violence":
            code = _sexual_violence(signals, decision_temporal)
        elif category == "domestic_violence":
            code = _domestic_violence(signals, decision_temporal)
        elif category == "child_abuse_or_endangerment":
            code = _child_abuse(signals, decision_temporal)
        elif category == "acute_crisis":
            code = _acute_crisis(signals, decision_temporal, set(topic_tags))
        else:
            code = _harm_others(signals, decision_temporal)
        codes.append(code)
    return tuple(sorted(set(codes), key=lambda code: (SITUATION_PRIORITY[code], code.value)))


def _active(signals: Sequence[RuleSignal]) -> list[RuleSignal]:
    return [signal for signal in signals if not signal.suppressed]


def _only_negated(signals: Sequence[RuleSignal]) -> bool:
    return bool(signals) and not _active(signals) and any(
        signal.suppression_reason == "local_negation"
        or signal.kind in {"negated_disclosure", "denied_disclosure"}
        for signal in signals
    )


def _effective_temporal(signal: RuleSignal, fallback: TemporalStatus) -> TemporalStatus:
    return signal.temporal if signal.temporal != TemporalStatus.UNSPECIFIED else fallback


def _has(
    signals: Sequence[RuleSignal],
    kinds: set[str],
    fallback: TemporalStatus,
    times: set[TemporalStatus] | None = None,
) -> bool:
    return any(
        signal.kind in kinds
        and (times is None or _effective_temporal(signal, fallback) in times)
        for signal in _active(signals)
    )


def _all_past(signals: Sequence[RuleSignal], fallback: TemporalStatus) -> bool:
    active = _active(signals)
    return bool(active) and all(
        _effective_temporal(signal, fallback) == TemporalStatus.PAST for signal in active
    )


def _suicide(
    signals: Sequence[RuleSignal],
    all_signals: Sequence[RuleSignal],
    temporal: TemporalStatus,
) -> SituationCode:
    immediate = {TemporalStatus.CURRENT, TemporalStatus.IMMINENT}
    overdose = _has(
        all_signals,
        {"possible_medical_emergency", "overdose", "poisoning", "in_progress_attempt"},
        temporal,
        immediate,
    )
    if overdose or _has(signals, {"in_progress", "attempt_in_progress"}, temporal, immediate):
        return SituationCode.SUICIDE_ATTEMPT_OR_IN_PROGRESS
    if _has(
        signals,
        {"explicit_intent", "plan", "means_available", "contextual_slang_intent"},
        temporal,
        immediate,
    ):
        return SituationCode.SUICIDE_ACTIVE_INTENT_OR_PLAN
    if _has(signals, {"preparation", "farewell", "sudden_calm", "means_available"}, temporal):
        return SituationCode.SUICIDE_PREPARATORY_SIGNALS
    if _has(
        signals,
        {"ideation_disclosure", "contextual_slang_ideation"},
        temporal,
        {TemporalStatus.CURRENT, TemporalStatus.RECENT},
    ):
        return SituationCode.SUICIDE_ACTIVE_IDEATION_UNSPECIFIED_INTENT
    if _has(signals, {"passive_ideation"}, temporal):
        return SituationCode.SUICIDE_PASSIVE_DEATH_WISH
    if _only_negated(signals):
        return SituationCode.SUICIDE_NEGATED_OR_DENIED
    if temporal == TemporalStatus.PAST or _all_past(signals, temporal):
        return SituationCode.SUICIDE_HISTORICAL
    return SituationCode.SUICIDE_SIGNAL_UNSPECIFIED


def _self_harm(signals: Sequence[RuleSignal], temporal: TemporalStatus) -> SituationCode:
    if _has(
        signals,
        {"in_progress", "contextual_slang_in_progress"},
        temporal,
        {TemporalStatus.CURRENT, TemporalStatus.IMMINENT},
    ):
        return SituationCode.SELF_HARM_IN_PROGRESS
    if _has(
        signals,
        {"direct_disclosure", "ongoing_disclosure"},
        temporal,
        {TemporalStatus.CURRENT, TemporalStatus.RECENT},
    ):
        return SituationCode.SELF_HARM_RECENT_DISCLOSURE
    if _has(
        signals,
        {"contextual_abbreviation", "contextual_slang_disclosure", "urge", "relapse"},
        temporal,
    ):
        return SituationCode.SELF_HARM_URGE_OR_RELAPSE
    if _only_negated(signals):
        return SituationCode.SELF_HARM_NEGATED_OR_DENIED
    if temporal == TemporalStatus.PAST or _all_past(signals, temporal):
        return SituationCode.SELF_HARM_HISTORICAL
    return SituationCode.SELF_HARM_SIGNAL_UNSPECIFIED


def _sexual_violence(signals: Sequence[RuleSignal], temporal: TemporalStatus) -> SituationCode:
    if _has(
        signals,
        {"ongoing_disclosure", "in_progress"},
        temporal,
        {TemporalStatus.CURRENT, TemporalStatus.IMMINENT},
    ):
        return SituationCode.SEXUAL_VIOLENCE_CURRENT_DANGER
    if _has(
        signals,
        {"direct_disclosure", "recent_disclosure"},
        temporal,
        {TemporalStatus.CURRENT, TemporalStatus.RECENT},
    ):
        return SituationCode.SEXUAL_VIOLENCE_RECENT
    if temporal == TemporalStatus.PAST or _all_past(signals, temporal):
        return SituationCode.SEXUAL_VIOLENCE_HISTORICAL_TRAUMA
    return SituationCode.SEXUAL_VIOLENCE_SIGNAL_UNSPECIFIED


def _domestic_violence(signals: Sequence[RuleSignal], temporal: TemporalStatus) -> SituationCode:
    if _has(
        signals,
        {"ongoing_disclosure", "in_progress", "current_threat"},
        temporal,
        {TemporalStatus.CURRENT, TemporalStatus.IMMINENT},
    ):
        return SituationCode.DOMESTIC_VIOLENCE_CURRENT_DANGER
    if _has(
        signals,
        {"direct_disclosure", "recent_disclosure"},
        temporal,
        {TemporalStatus.CURRENT, TemporalStatus.RECENT},
    ):
        return SituationCode.DOMESTIC_VIOLENCE_RECENT
    if temporal == TemporalStatus.PAST or _all_past(signals, temporal):
        return SituationCode.DOMESTIC_VIOLENCE_HISTORICAL
    return SituationCode.DOMESTIC_VIOLENCE_SIGNAL_UNSPECIFIED


def _child_abuse(signals: Sequence[RuleSignal], temporal: TemporalStatus) -> SituationCode:
    if _has(
        signals,
        {"ongoing_disclosure", "in_progress", "current_threat"},
        temporal,
        {TemporalStatus.CURRENT, TemporalStatus.IMMINENT},
    ):
        return SituationCode.CHILD_ABUSE_CURRENT_DANGER
    if _has(
        signals,
        {"third_party_disclosure", "third_party_endangerment", "reported_endangerment"},
        temporal,
        {TemporalStatus.CURRENT, TemporalStatus.IMMINENT, TemporalStatus.RECENT},
    ):
        return SituationCode.CHILD_ENDANGERMENT_THIRD_PARTY
    if _has(signals, {"historical_disclosure"}, temporal) or temporal == TemporalStatus.PAST:
        return SituationCode.CHILDHOOD_ABUSE_HISTORICAL_TRAUMA
    return SituationCode.CHILD_ABUSE_SIGNAL_UNSPECIFIED


def _acute_crisis(
    signals: Sequence[RuleSignal],
    temporal: TemporalStatus,
    topic_tags: set[str],
) -> SituationCode:
    if _has(
        signals,
        {"possible_medical_emergency", "overdose", "poisoning"},
        temporal,
        {TemporalStatus.CURRENT, TemporalStatus.IMMINENT, TemporalStatus.RECENT},
    ):
        return SituationCode.POSSIBLE_OVERDOSE_OR_POISONING
    if _has(
        signals,
        {"acute_distress", "panic"},
        temporal,
        {TemporalStatus.CURRENT, TemporalStatus.IMMINENT},
    ):
        return SituationCode.PANIC_OR_ACUTE_DISTRESS_CURRENT
    if "substance_use" in topic_tags or _has(
        signals,
        {"withdrawal", "acute_substance_concern", "substance_concern"},
        temporal,
    ):
        return SituationCode.SUBSTANCE_WITHDRAWAL_OR_ACUTE_SUBSTANCE_CONCERN
    return SituationCode.ACUTE_CRISIS_SIGNAL_UNSPECIFIED


def _harm_others(signals: Sequence[RuleSignal], temporal: TemporalStatus) -> SituationCode:
    if _has(
        signals,
        {"explicit_intent", "plan", "means_available", "contextual_slang_intent"},
        temporal,
        {TemporalStatus.CURRENT, TemporalStatus.IMMINENT},
    ):
        return SituationCode.HARM_OTHERS_ACTIVE_INTENT
    return SituationCode.HARM_OTHERS_SIGNAL_UNSPECIFIED

