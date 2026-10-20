"""Bounded, content-free recent safety state.

This state is an application-policy aid, not a clinical score.  It stores no
message text, spans, probabilities, identity, or location.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import CrisisTier, PolicyStateUpdate, Severity, TemporalStatus


STATE_VERSION = "recent-safety-state/1.0"
MAX_OBSERVATIONS = 8
MAX_AGE_TURNS = 12
RESOURCE_REPEAT_INTERVAL_TURNS = 3

_TIER_RANK = {CrisisTier.NORMAL: 0, CrisisTier.MONITORED: 1, CrisisTier.CRISIS: 2}
_SEVERITY_RANK = {
    Severity.NONE: 0,
    Severity.WATCH: 1,
    Severity.ELEVATED: 2,
    Severity.URGENT: 3,
}


@dataclass(frozen=True, slots=True)
class SafetyObservation:
    turn_index: int
    categories: tuple[str, ...]
    tier: CrisisTier
    severity: Severity
    temporal_status: TemporalStatus
    risk_signal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "categories": list(self.categories),
            "tier": self.tier.value,
            "severity": self.severity.value,
            "temporal_status": self.temporal_status.value,
            "risk_signal": self.risk_signal,
        }


@dataclass(frozen=True, slots=True)
class RecentSafetyState:
    state_version: str = STATE_VERSION
    recent_signal_turns: int = 0
    recent_categories: tuple[str, ...] = ()
    highest_recent_tier: CrisisTier = CrisisTier.NORMAL
    highest_recent_severity: Severity = Severity.NONE
    current_or_imminent_seen: bool = False
    direct_safety_question_pending: bool = False
    resources_last_shown_turn: int | None = None
    escalation_trend: str = "none"
    last_updated_turn: int = -1
    observations: tuple[SafetyObservation, ...] = ()
    classified_crisis_messages: int = 0

    def __post_init__(self) -> None:
        if self.state_version != STATE_VERSION:
            raise ValueError(f"Unsupported state schema: {self.state_version!r}")
        if len(self.observations) > MAX_OBSERVATIONS:
            raise ValueError("Recent safety state exceeds its observation bound")
        if self.escalation_trend not in {"none", "new", "increasing", "decreasing", "stable"}:
            raise ValueError("Invalid escalation_trend")
        if self.last_updated_turn < -1:
            raise ValueError("last_updated_turn cannot be below -1")
        _nonnegative_int(self.classified_crisis_messages, 'classified_crisis_messages')
        if any(obs.turn_index > self.last_updated_turn for obs in self.observations):
            raise ValueError("Observation turn exceeds last_updated_turn")

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "recent_signal_turns": self.recent_signal_turns,
            "recent_categories": list(self.recent_categories),
            "highest_recent_tier": self.highest_recent_tier.value,
            "highest_recent_severity": self.highest_recent_severity.value,
            "current_or_imminent_seen": self.current_or_imminent_seen,
            "direct_safety_question_pending": self.direct_safety_question_pending,
            "resources_last_shown_turn": self.resources_last_shown_turn,
            "escalation_trend": self.escalation_trend,
            "last_updated_turn": self.last_updated_turn,
            "observations": [observation.to_dict() for observation in self.observations],
            "classified_crisis_messages": self.classified_crisis_messages,
        }

    def conversation_indicator(self) -> dict[str, Any]:
        """User-requested >3-message annotation, NEVER a clinical/routing rule."""
        return {'classified_crisis_messages':self.classified_crisis_messages,
            'threshold_messages':4,
            'conversation_count_label':'crisis' if self.classified_crisis_messages > 3 else 'normal',
            'semantics':'Count annotation only; not a diagnosis, safety assessment or routing input.',
            'single_urgent_message_handled_immediately':True}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecentSafetyState":
        """Restore only the documented content-free fields."""

        if value.get("state_version") != STATE_VERSION:
            raise ValueError("Unsupported or missing recent safety state version")
        raw_observations = value.get("observations", ())
        if not isinstance(raw_observations, list):
            raise ValueError("observations must be a list")
        observations: list[SafetyObservation] = []
        for raw in raw_observations[-MAX_OBSERVATIONS:]:
            if not isinstance(raw, Mapping):
                raise ValueError("Invalid safety observation")
            observations.append(SafetyObservation(
                turn_index=_nonnegative_int(raw.get("turn_index"), "turn_index"),
                categories=_string_tuple(raw.get("categories", ())),
                tier=CrisisTier(raw.get("tier")),
                severity=Severity(raw.get("severity")),
                temporal_status=TemporalStatus(raw.get("temporal_status")),
                risk_signal=bool(raw.get("risk_signal")),
            ))
        resource_turn = value.get("resources_last_shown_turn")
        if resource_turn is not None:
            resource_turn = _nonnegative_int(resource_turn, "resources_last_shown_turn")
        return cls(
            recent_signal_turns=_nonnegative_int(value.get("recent_signal_turns", 0), "recent_signal_turns"),
            recent_categories=_string_tuple(value.get("recent_categories", ())),
            highest_recent_tier=CrisisTier(value.get("highest_recent_tier", "normal")),
            highest_recent_severity=Severity(value.get("highest_recent_severity", "none")),
            current_or_imminent_seen=bool(value.get("current_or_imminent_seen")),
            direct_safety_question_pending=bool(value.get("direct_safety_question_pending")),
            resources_last_shown_turn=resource_turn,
            escalation_trend=str(value.get("escalation_trend", "none")),
            last_updated_turn=int(value.get("last_updated_turn", -1)),
            observations=tuple(observations),
            classified_crisis_messages=_nonnegative_int(value.get('classified_crisis_messages',0),
                'classified_crisis_messages'),
        )


def update_recent_safety_state(
    state: RecentSafetyState,
    update: PolicyStateUpdate,
    *,
    turn_index: int,
    resources_shown: bool = False,
    safety_question_answered: bool = False,
    message_classified_crisis: bool = False,
) -> RecentSafetyState:
    """Advance state with a bounded observation and deterministic decay."""

    if isinstance(turn_index, bool) or not isinstance(turn_index, int) or turn_index < 0:
        raise ValueError("turn_index must be a non-negative integer")
    if turn_index <= state.last_updated_turn:
        raise ValueError("turn_index must increase monotonically")
    if type(message_classified_crisis) is not bool:
        raise TypeError('message_classified_crisis must be a boolean')

    minimum_turn = max(0, turn_index - MAX_AGE_TURNS)
    retained = [
        observation for observation in state.observations
        if observation.turn_index >= minimum_turn
    ]
    observation = SafetyObservation(
        turn_index=turn_index,
        categories=tuple(dict.fromkeys(update.categories)),
        tier=update.highest_turn_tier,
        severity=update.highest_turn_severity,
        temporal_status=update.temporal_status,
        risk_signal=update.risk_signal_turn,
    )
    previous = retained[-1] if retained else None
    retained.append(observation)
    retained = retained[-MAX_OBSERVATIONS:]

    signal_observations = [item for item in retained if item.risk_signal]
    categories = sorted({
        category for item in signal_observations for category in item.categories
    })
    highest_tier = max(
        (item.tier for item in signal_observations),
        key=_TIER_RANK.__getitem__,
        default=CrisisTier.NORMAL,
    )
    highest_severity = max(
        (item.severity for item in signal_observations),
        key=_SEVERITY_RANK.__getitem__,
        default=Severity.NONE,
    )
    immediate_seen = any(
        item.risk_signal
        and item.temporal_status in {TemporalStatus.CURRENT, TemporalStatus.IMMINENT}
        for item in retained
    )
    pending = update.direct_safety_question_requested or (
        state.direct_safety_question_pending and not safety_question_answered
    )
    resource_turn = turn_index if resources_shown else state.resources_last_shown_turn

    return RecentSafetyState(
        recent_signal_turns=len(signal_observations),
        recent_categories=tuple(categories),
        highest_recent_tier=highest_tier,
        highest_recent_severity=highest_severity,
        current_or_imminent_seen=immediate_seen,
        direct_safety_question_pending=pending,
        resources_last_shown_turn=resource_turn,
        escalation_trend=_trend(previous, observation),
        last_updated_turn=turn_index,
        observations=tuple(retained),
        classified_crisis_messages=state.classified_crisis_messages + int(message_classified_crisis),
    )


def should_repeat_resources(
    state: RecentSafetyState,
    *,
    current_turn: int,
    danger_escalated: bool = False,
    location_changed: bool = False,
) -> bool:
    """Decide resource-block repetition without ever suppressing first display."""

    if current_turn < 0:
        raise ValueError("current_turn must be non-negative")
    if state.resources_last_shown_turn is None:
        return True
    if danger_escalated or location_changed:
        return True
    return current_turn - state.resources_last_shown_turn >= RESOURCE_REPEAT_INTERVAL_TURNS


def _trend(previous: SafetyObservation | None, current: SafetyObservation) -> str:
    current_rank = (_TIER_RANK[current.tier], _SEVERITY_RANK[current.severity])
    if previous is None:
        return "new" if current.risk_signal else "none"
    previous_rank = (_TIER_RANK[previous.tier], _SEVERITY_RANK[previous.severity])
    if current_rank > previous_rank:
        return "increasing"
    if current_rank < previous_rank:
        return "decreasing"
    return "stable" if current.risk_signal or previous.risk_signal else "none"


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError("Expected a list of strings")
    return tuple(dict.fromkeys(value))

