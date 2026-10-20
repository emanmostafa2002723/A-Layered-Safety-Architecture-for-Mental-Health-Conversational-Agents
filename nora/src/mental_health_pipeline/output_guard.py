"""Buffered output-safety contract and conservative development guard.

``PlaceholderOutputGuard`` is deliberately *not* marked production-ready.  It
provides deterministic defence in depth while a separately evaluated output
classifier is built, but it must never be represented as that classifier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final, Protocol, runtime_checkable


PLACEHOLDER_OUTPUT_GUARD_VERSION: Final = "supplemental-rules-1.4.0"
FIXED_BLOCKED_RESPONSE: Final = (
    "I’m sorry, but I can’t provide that response safely. If you or someone else "
    "may be in immediate danger, contact the local emergency number or go to the "
    "nearest emergency department now. If it is safe to do so, reach out to a "
    "trusted person or qualified crisis professional for immediate support."
)


@dataclass(frozen=True, slots=True)
class OutputGuardContext:
    """Policy facts the guard must confirm in the complete response.

    ``required_phrases`` are all mandatory.  Each tuple in
    ``required_any_groups`` represents alternatives, of which at least one must
    appear.  Comparisons are whitespace-normalized and case-insensitive.
    """

    route: str = "standard"
    required_phrases: tuple[str, ...] = ()
    required_any_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.route.strip():
            raise ValueError("route must not be blank")
        if any(not phrase.strip() for phrase in self.required_phrases):
            raise ValueError("required_phrases must not contain blank values")
        for group in self.required_any_groups:
            if not group or any(not phrase.strip() for phrase in group):
                raise ValueError("required_any_groups must contain non-blank alternatives")


@dataclass(frozen=True, slots=True)
class OutputGuardDecision:
    approved: bool
    response: str
    violations: tuple[str, ...]
    guard_name: str
    guard_version: str
    production_ready: bool
    checked_character_count: int

    @property
    def intervened(self) -> bool:
        return not self.approved

    def to_dict(self) -> dict[str, object]:
        return {
            "approved": self.approved,
            "response": self.response,
            "violations": list(self.violations),
            "guard_name": self.guard_name,
            "guard_version": self.guard_version,
            "production_ready": self.production_ready,
            "checked_character_count": self.checked_character_count,
        }


@runtime_checkable
class OutputGuard(Protocol):
    """Contract for a guard that evaluates one complete, buffered response."""

    name: str
    version: str
    production_ready: bool

    def evaluate(
        self,
        response: str,
        *,
        context: OutputGuardContext | None = None,
    ) -> OutputGuardDecision: ...


_INTERNAL_POLICY_PATTERNS: Final = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:system|developer)\s+(?:message|prompt|instruction)s?\b",
        r"\bresponse[ _-]?policy(?:\s+packet)?\b",
        r"\binternal\s+(?:policy|instructions?|chain[- ]of[- ]thought)\b",
        r"\b(?:the\s+app|the\s+application)\s+(?:supplies|owns)\s+(?:the\s+)?(?:exact\s+)?(?:emergency\s+contacts|fixed\s+contacts|fixed\s+notices)\b",
        r"\b(?:follow\s+the\s+trusted\s+policy|untrusted\s+reference\s+data)\b",
        r"\bI\s+(?:was|am)\s+instructed\s+to\b",
        r"<\|(?:system|assistant|user)\|>",
        r"\[/?INST\]",
        r"<<\s*/?SYS\s*>>",
    )
)

_PERSONALIZED_MEDICATION_PATTERNS: Final = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:you\s+(?:should|must|need\s+to)|I\s+(?:recommend|advise)\s+(?:that\s+)?you)\s+"
        r"(?:start|stop|increase|decrease|double|halve|skip|switch|take)\b",
        r"\b(?:increase|decrease|double|halve|skip)\s+your\s+(?:dose|dosage|medication)\b",
        r"\b(?:take|use)\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|milligrams?|micrograms?)\b",
        r"\bI\s+(?:prescribe|am\s+prescribing)\b",
    )
)

_HARM_DETAIL_PATTERNS: Final = tuple(
    re.compile(pattern, re.IGNORECASE | re.DOTALL)
    for pattern in (
        r"\bhow\s+to\s+(?:kill\s+(?:yourself|someone)|die\s+by\s+suicide|self[- ]?harm)\b",
        r"\bstep[- ]by[- ]step\s+(?:guide|instructions?)\s+(?:for|to)\s+"
        r"(?:suicide|self[- ]?harm|kill)",
        r"\b(?:most|more)\s+(?:effective|lethal|painless|reliable)\s+"
        r"(?:suicide|self[- ]?harm)\s+method\b",
        r"\blethal\s+(?:dose|dosage)\b.{0,80}\b\d+(?:\.\d+)?\s*"
        r"(?:mg|mcg|g|milligrams?|micrograms?)\b",
        r"\b(?:instructions?|directions?)\s+for\s+(?:making|building)\b.{0,80}"
        r"\b(?:weapon|poison|explosive)\b",
    )
)

_PROFESSIONAL_IMPERSONATION_PATTERNS: Final = tuple(
    re.compile(pattern, re.IGNORECASE | re.MULTILINE)
    for pattern in (
        r"\bI(?:\s+am|['’]m)\s+(?!not\b)(?:an?\s+)?"
        r"(?:(?:licensed|certified|registered|board[- ]certified)\s+)?"
        r"(?:counsel(?:or|lor)|therapist|psychologist|psychiatrist|doctor|physician|"
        r"clinician|social\s+worker|mental[- ]health\s+professional)\b",
        r"\bAs\s+(?:an?\s+)?(?:(?:licensed|certified|registered|board[- ]certified)\s+)?"
        r"(?:counsel(?:or|lor)|therapist|psychologist|psychiatrist|doctor|physician|"
        r"clinician|social\s+worker|mental[- ]health\s+professional)\b",
        r"\bI\s+speciali[sz]e\s+in\b",
        r"\b(?:my\s+rapist|my\s+therapist|I\s+(?:was\s+diagnosed|experienced\s+abuse)|when\s+I\s+was\s+a\s+child)\b",
        r"\bI\s+have\s+(?:been\s+)?(?:worked|working|practi[cs]ed|practi[cs]ing)\s+"
        r"(?:in|as)\b",
        r"\bI\s+(?:hold|have)\s+(?:an?\s+)?(?:DBH|LPC|NCC|LMFT|LCSW|PsyD|MD)\b",
        r"^(?:best\s+wishes|sincerely|warm\s+regards|regards),?\s+"
        r"[A-Z][A-Za-z.'’-]+(?:\s+[A-Z][A-Za-z.'’-]+){0,3},?\s+"
        r"(?:DBH|LPC|NCC|LMFT|LCSW|PsyD|MD)(?:\s*,\s*(?:DBH|LPC|NCC|LMFT|LCSW|PsyD|MD))*\s*$",
    )
)

_UNSUPPORTED_CLINICAL_INFERENCE_PATTERNS: Final = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"(?<!said )(?<!told\sme\s)\byou\s+"
        r"(?:are|feel|seem|sound|may\s+be|might\s+be|must\s+be)\s+"
        r"(?:clinically\s+)?(?:depressed|traumati[sz]ed|suicidal|manic|"
        r"psychotic|bipolar|dissociative)\b",
        r"\byou\s+(?:have|suffer\s+from|are\s+experiencing)\s+"
        r"(?:an?\s+)?"
        r"(?:depression|anxiety|PTSD|post[- ]traumatic\s+stress|bipolar\s+disorder|"
        r"psychosis|a\s+panic\s+disorder|a\s+personality\s+disorder)\b",
        r"\byou\s+(?:are|feel|seem|sound|may\s+be|might\s+be|must\s+be)\s+clinically\s+anxious\b",
        r"\byour\s+(?:depression|PTSD|post[- ]traumatic\s+stress|trauma|"
        r"bipolar\s+disorder|psychosis|panic\s+disorder|personality\s+disorder)\b",
        r"\bwhat\s+you(?:['’]re|\s+are)\s+experiencing\s+is\s+"
        r"(?:depression|anxiety|PTSD|post[- ]traumatic\s+stress|bipolar\s+disorder|"
        r"psychosis|a\s+panic\s+disorder|a\s+personality\s+disorder)\b",
    )
)


def immediate_safety_question_present(text: str) -> bool:
    """Target direct present-safety/injury questions, not any use of 'safe'.

    Still a narrow supplemental check, not a semantic/clinical assessment.
    In particular 'if you feel safe, can you call someone?' is not an answer
    to whether the person might act now or already needs help for an injury.
    """
    patterns = (
        r'\b(?:are|do|might|may|could|will) you\b[^?]{0,80}\b(?:safe|danger|act(?:ing)? on|harm(?:ing)? yourself|hurt(?:ing)? yourself|end your life|kill yourself)\b',
        r'\b(?:can|could|will|would) you (?:stay|keep yourself) safe\b',
        r'\b(?:have you|whether you have)\b[^?]{0,60}\b(?:hurt|harmed|injured) yourself\b',
        r'\bwhether you (?:might|may|will|could) act\b',
        r'\b(?:have you|do you have|whether you have)\b[^?]{0,35}\b(?:plan|intention) to (?:act|carry out|harm|hurt|kill|end)\b',
    )
    questions = re.findall(r'[^.!?]*\?', text.casefold())
    return any(re.search(pattern, question) for question in questions for pattern in patterns)


def _normalized_for_match(value: str) -> str:
    return " ".join(value.casefold().split())


class PlaceholderOutputGuard:
    """Deterministic stopgap that always reports itself as non-production.

    The input must be a single string containing the whole generated response;
    token streams, iterators, and partial chunks are rejected.  Any violation
    returns the same fixed response, so rejected model text cannot leak through
    error paths.
    """

    name = "placeholder_output_guard"
    version = PLACEHOLDER_OUTPUT_GUARD_VERSION
    production_ready = False

    def __init__(self, *, max_characters: int = 16_000) -> None:
        if max_characters < 256:
            raise ValueError("max_characters must be at least 256")
        self.max_characters = max_characters

    def evaluate(
        self,
        response: str,
        *,
        context: OutputGuardContext | None = None,
    ) -> OutputGuardDecision:
        guard_context = context or OutputGuardContext()
        violations: list[str] = []

        if not isinstance(response, str):
            return self._blocked(("invalid_response_type",), 0)

        checked_count = len(response)
        stripped = response.strip()
        if not stripped:
            violations.append("empty_response")
        if checked_count > self.max_characters:
            violations.append("response_too_long")

        if any(pattern.search(response) for pattern in _INTERNAL_POLICY_PATTERNS):
            violations.append("internal_policy_leakage")
        if any(pattern.search(response) for pattern in _PERSONALIZED_MEDICATION_PATTERNS):
            violations.append("personalized_medication_advice")
        if any(pattern.search(response) for pattern in _HARM_DETAIL_PATTERNS):
            violations.append("harmful_procedural_detail")
        if any(pattern.search(response) for pattern in _PROFESSIONAL_IMPERSONATION_PATTERNS):
            violations.append("professional_identity_or_credentials")
        if any(pattern.search(response) for pattern in _UNSUPPORTED_CLINICAL_INFERENCE_PATTERNS):
            violations.append("unsupported_clinical_inference")

        normalized_response = _normalized_for_match(response)
        if any(
            _normalized_for_match(phrase) not in normalized_response
            for phrase in guard_context.required_phrases
        ):
            violations.append("missing_required_content")

        if any(
            not any(_normalized_for_match(option) in normalized_response for option in group)
            for group in guard_context.required_any_groups
        ):
            violations.append("missing_required_alternative")

        if violations:
            # Preserve stable order while collapsing duplicate categories.
            return self._blocked(tuple(dict.fromkeys(violations)), checked_count)

        return OutputGuardDecision(
            approved=True,
            response=response,
            violations=(),
            guard_name=self.name,
            guard_version=self.version,
            production_ready=self.production_ready,
            checked_character_count=checked_count,
        )

    def inspect(self, candidate_response: str, policy: Any) -> Any:
        """Adapt to ``response_policy.output_contract.OutputGuard``.

        The import is lazy so the standalone guard remains usable without
        importing the policy package. Required visible elements are symbolic,
        allowlisted policy identifiers; an unknown identifier fails closed.
        """

        from .response_policy.output_contract import (
            OutputGuardAction as PolicyGuardAction,
            OutputGuardDecision as PolicyGuardDecision,
        )

        decision = self.evaluate(candidate_response)
        reasons = list(decision.violations)
        normalized = _normalized_for_match(candidate_response) if isinstance(candidate_response, str) else ""
        required = getattr(policy, "required_visible_elements", ())
        if not isinstance(required, tuple) or not all(isinstance(item, str) for item in required):
            reasons.append("invalid_required_visible_elements")
        else:
            for marker in required:
                if not _required_visible_element_present(marker, normalized):
                    reasons.append("missing_required_visible_element")
                    break

        action = PolicyGuardAction.APPROVE if not reasons else PolicyGuardAction.FALLBACK
        return PolicyGuardDecision(
            action=action,
            guard_version=self.version,
            policy_version=str(getattr(policy, "policy_version", "")),
            output_guard_profile=str(getattr(policy, "output_guard_profile", "")),
            reason_codes=tuple(dict.fromkeys(reasons)),
            evaluated_complete_response=True,
        )

    def _blocked(
        self,
        violations: tuple[str, ...],
        checked_character_count: int,
    ) -> OutputGuardDecision:
        return OutputGuardDecision(
            approved=False,
            response=FIXED_BLOCKED_RESPONSE,
            violations=violations,
            guard_name=self.name,
            guard_version=self.version,
            production_ready=self.production_ready,
            checked_character_count=checked_character_count,
        )

    def inspect_with_context(self, candidate_response: str, policy: Any, *, user_text: str) -> Any:
        """Check known dialogue failures in addition to the existing rule guard.

        These targeted checks are not a clinical classifier. They trigger one
        buffered rewrite and then the ordinary guarded fallback if still unmet.
        """
        from .response_policy.output_contract import OutputGuardAction, OutputGuardDecision
        decision = self.inspect(candidate_response, policy)
        if decision.action != OutputGuardAction.APPROVE:
            return decision
        body = candidate_response.split("\n\n", 1)[-1].strip()
        normalized = _normalized_for_match(body)
        reasons = []
        trivial = normalized.rstrip(".! ") in {"ok", "okay", "acknowledged", "i'm sorry to hear that", "i am sorry to hear that", "you are safe"}
        if trivial or normalized.startswith("the user is "):
            reasons.append("insufficient_supportive_content")
        text = _normalized_for_match(user_text)
        suicide = "suicidal_ideation" in getattr(policy, "crisis_labels", ())
        if suicide and current_first_person_safety_disclosure(text):
            if not immediate_safety_question_present(body):
                reasons.append("missing_immediate_safety_question")
        if not reasons:
            return decision
        return OutputGuardDecision(action=OutputGuardAction.RETRY,
            guard_version=self.version, policy_version=policy.policy_version,
            output_guard_profile=policy.output_guard_profile,
            reason_codes=tuple(reasons), evaluated_complete_response=True)

    def readiness(self) -> dict[str, object]:
        return {
            "ready": False,
            "component": self.name,
            "version": self.version,
            "reason": "placeholder_output_guard_requires_replacement",
        }


def current_first_person_safety_disclosure(text: str) -> bool:
    """Targeted regression coverage, NOT a validated contextual risk assessor.

    Local clause checks prevent a denial/history in another clause from erasing
    a positive current disclosure. Quoted/educational/third-party mentions alone
    must not trigger a question about the user's own immediate intent.
    """
    text = _normalized_for_match(text).replace('’', "'")
    # Quoted examples are untrusted mentions, not the speaker's own disclosure.
    text = re.sub(r'"[^"\n]*"|“[^”\n]*”', '', text)
    target = re.compile(r"\b(?:kill(?:ing)? myself|end(?:ing)? my life|hurt myself|"
                        r"suicidal thoughts|thoughts (?:about|of) suicide|thinking (?:about|of) suicide)\b")
    for clause in re.split(r'[.!?;]|\b(?:but|however)\b', text):
        for match in target.finditer(clause):
            before = clause[:match.start()]
            if not re.search(r"\bi(?:'m| am| have| had| keep| feel| want| plan| might| may| will| was| do| don't| can| can't)\b", before):
                continue
            window = clause[max(0, match.start() - 60):match.end() + 40]
            if re.search(r"\b(?:years? ago|in the past|used to|previously)\b", window):
                continue
            if re.search(r"\b(?:don't|do not|not|never|no longer)\b[^,;.!?]{0,35}$", before):
                continue
            if re.search(r"\b(?:researching|studying|writing|explaining|translating|what does|what is)\b", before):
                continue
            return True
    return False


_REQUIRED_VISIBLE_ALTERNATIVES: Final[dict[str, tuple[str, ...]]] = {
    "worldwide_emergency_fallback": (
        "local emergency services",
        "local emergency number",
        "nearest emergency department",
        "findahelpline.com",
    ),
    "local_resources_if_eligible": (
        "local emergency services",
        "local emergency number",
        "findahelpline.com",
        "9-1-1",
        "911",
        "9-8-8",
        "988",
    ),
    "boundary_limit_notice": (
        "i can’t diagnose",
        "i can't diagnose",
        "i cannot diagnose",
        "i can’t provide a personalized",
        "i can't provide a personalized",
        "i cannot provide a personalized",
        "general information",
    ),
    "scope_limit_notice": (
        "english-speaking adults",
        "adults age 18",
        "age 18 or older",
        "outside the supported scope",
    ),
    'scope_uncertainty_flag':(
        'how old are you', 'are you an adult', 'are you under 18',
        'are you asking about yourself or someone else',
        'is a child involved', 'is this about a child',
    ),
}


def _required_visible_element_present(marker: str, normalized_response: str) -> bool:
    alternatives = _REQUIRED_VISIBLE_ALTERNATIVES.get(marker)
    if alternatives is None:
        # Unknown symbolic requirements are not silently ignored.
        return False
    return any(_normalized_for_match(value) in normalized_response for value in alternatives)


__all__ = [
    "FIXED_BLOCKED_RESPONSE",
    "OutputGuard",
    "OutputGuardContext",
    "OutputGuardDecision",
    "PLACEHOLDER_OUTPUT_GUARD_VERSION",
    "PlaceholderOutputGuard",
]
