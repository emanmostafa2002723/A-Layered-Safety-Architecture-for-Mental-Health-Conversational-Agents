"""The single rule/regex channel for the input guardrail.

Rules favor multi-word, first-person or direct-disclosure patterns over bare
keywords.  Bare topic words (for example, ``suicide`` in an academic sentence)
must not independently create a crisis route.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from .schemas import RuleExtraction, SafetyMatch, Severity, Span, TemporalStatus


@dataclass(frozen=True)
class Rule:
    rule_id: str
    pattern: re.Pattern[str]
    category: str | None
    kind: str
    severity: Severity
    implied_temporal: TemporalStatus = TemporalStatus.UNSPECIFIED
    confidence: str = "high"


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.I)


SAFETY_RULES = (
    Rule("SI_NEGATED_INTENT", _compile(
        r"\b(?:i\s+am|i['’]m)\s+(?:not|never)\s+(?:going|planning)\s+to\s+"
        r"(?:kill\s+myself|end\s+my\s+life|take\s+my\s+own\s+life)\b|"
        r"\bi\s+(?:don['’]?t|do\s+not)\s+(?:want|intend|plan)\s+to\s+"
        r"(?:kill\s+myself|end\s+my\s+life|take\s+my\s+own\s+life)\b|"
        r"\bi\s+(?:do\s+not|don['’]?t)\s+have\s+a\s+plan\s+to\s+"
        r"(?:kill\s+myself|end\s+my\s+life|take\s+my\s+own\s+life)\b|"
        r"\bi\s+have\s+no\s+plan\s+to\s+"
        r"(?:kill\s+myself|end\s+my\s+life|take\s+my\s+own\s+life)\b|"
        r"\bi\s+no\s+longer\s+(?:want|intend|plan)\s+to\s+"
        r"(?:kill\s+myself|end\s+my\s+life|take\s+my\s+own\s+life)\b|"
        r"\bi(?:'ve|\s+have)?\s*decided\s+not\s+to\s+"
        r"(?:kill\s+myself|end\s+my\s+life|take\s+my\s+own\s+life)\b|"
        r"\bi(?:'m|\s+am)\s+not\s+(?:thinking|considering)\s+(?:about|of)\s+"
        r"(?:suicide|killing\s+myself|ending\s+my\s+life)\b|"
        r"\bi(?:'m|\s+am)\s+safe\s+and\s+not\s+(?:thinking|considering)\s+(?:about|of)\s+"
        r"(?:suicide|killing\s+myself|ending\s+my\s+life)\b|"
        r"\bi(?:'m|\s+am)\s+not\s+suicidal\b|"
        r"\bi\s+have\s+no\s+suicidal\s+thoughts\b"
    ), "suicidal_ideation", "negated_disclosure", Severity.ELEVATED),
    Rule("SELF_HARM_NEGATED", _compile(
        r"\bi\s+(?:won['’]?t|will\s+not|am\s+not\s+going\s+to|don['’]?t\s+want\s+to)\s+"
        r"(?:hurt|harm|cut|burn)\s+myself\b|"
        r"\bi(?:'m|\s+am)\s+not\s+(?:thinking|considering)\s+(?:about|of)\s+"
        r"(?:hurting|harming|cutting|burning)\s+myself\b|"
        r"\bi(?:'m|\s+am)\s+safe\s+and\s+not\s+(?:thinking|considering)\s+(?:about|of)\s+"
        r"(?:hurting|harming|cutting|burning)\s+myself\b|"
        r"\bi\s+have\s+no\s+(?:urge|urges)\s+to\s+(?:hurt|harm|cut|burn)\s+myself\b"
    ), "self_directed_harm", "negated_disclosure", Severity.ELEVATED),
    Rule("SI_DIRECT_INTENT", _compile(
        r"\b(?:i\s+(?:really\s+)?(?:want|intend|plan)\s+to|"
        r"i(?:'m|\s+am)\s+(?:going|planning)\s+to|"
        r"i\s+(?:have|made|worked\s+out)\s+a\s+plan\s+to|"
        r"i(?:'ve|\s+have)\s+made\s+a\s+plan\s+to|i\s+will)\s+"
        r"(?:kill\s+myself|end\s+my\s+life|take\s+my\s+own\s+life)\b"
    ), "suicidal_ideation", "explicit_intent", Severity.URGENT, TemporalStatus.IMMINENT),
    Rule("SI_EXPLICIT_THOUGHTS", _compile(
        r"\b(?:i(?:'m|\s+am|\s+have\s+been)\s+(?:thinking|having\s+thoughts)\s+(?:about|of)|"
        r"i\s+have)\s+(?:suicid(?:e|al\s+thoughts)|killing\s+myself|ending\s+my\s+life)\b"
    ), "suicidal_ideation", "ideation_disclosure", Severity.ELEVATED),
    # Youth algospeak is deliberately contextual. These expressions are not
    # treated as dangerous when they occur as bare words or unrelated phrases.
    Rule("SI_SLANG_UNALIVE", _compile(
        r"\b(?:i\s+(?:want|plan|intend)\s+to|i(?:'m|\s+am)\s+going\s+to|i\s+might)\s+"
        r"(?:unalive|delete|off)\s+myself\b"
    ), "suicidal_ideation", "contextual_slang_intent", Severity.URGENT,
         TemporalStatus.IMMINENT, "contextual"),
    Rule("SI_SLANG_KMS", _compile(
        r"\b(?:i(?:'m|\s+am)\s+(?:gonna|going\s+to)|i\s+(?:might|may|will|wanna|want\s+to))\s+kms\b"
    ), "suicidal_ideation", "contextual_slang_intent", Severity.URGENT,
         TemporalStatus.IMMINENT, "contextual"),
    Rule("SI_SLANG_13", _compile(
        r"\bi\s+(?:want|plan|intend)\s+to\s+13\s+myself\b|"
        r"\bi(?:'m|\s+am)\s+13(?:['’]?ing)\s+myself\b"
    ), "suicidal_ideation", "contextual_slang_intent", Severity.URGENT,
         TemporalStatus.IMMINENT, "contextual"),
    Rule("SI_SLANG_SEWER_SLIDE", _compile(
        r"\bi(?:'m|\s+am|\s+have\s+been)\s+(?:thinking|talking)\s+(?:about|of)\s+"
        r"(?:a\s+)?sewer[ -]?slide\b"
    ), "suicidal_ideation", "contextual_slang_ideation", Severity.ELEVATED,
         TemporalStatus.UNSPECIFIED, "contextual"),
    Rule("SI_PLAN_DATE", _compile(
        r"\b(?:i\s+have\s+)?(?:a\s+)?plan\s+and\s+(?:a\s+)?(?:date|day|time)\s+(?:picked|set|chosen)\b"
    ), "suicidal_ideation", "plan", Severity.URGENT, TemporalStatus.IMMINENT),
    Rule("SI_PASSIVE_DEATH_WISH", _compile(
        r"\bi\s+(?:wish|hope|want)\s+(?:i\s+)?(?:was\s+dead|were\s+dead|wouldn['’]?t\s+wake\s+up|"
        r"could\s+go\s+to\s+sleep\s+and\s+never\s+wake\s+up|could\s+disappear)\b"
    ), "suicidal_ideation", "passive_ideation", Severity.ELEVATED),
    Rule("SI_BETTER_OFF_WITHOUT_ME", _compile(
        r"\b(?:they|everyone|my\s+(?:family|friends|children|kids))\s+(?:would|wouldn['’]?t)\s+"
        r"be\s+better\s+off\s+without\s+me\b"
    ), "suicidal_ideation", "farewell", Severity.ELEVATED),
    Rule("SI_GIVING_AWAY_POSSESSIONS", _compile(
        r"\bi(?:'m|\s+am|\s+have\s+been)\s+giving\s+away\s+(?:all\s+)?my\s+(?:stuff|things|possessions)\b"
    ), "suicidal_ideation", "preparation", Severity.ELEVATED),
    Rule("SI_SUDDEN_CALM", _compile(
        r"\bi\s+(?:finally|suddenly)\s+feel\s+calm\s+about\s+(?:everything|it\s+all|my\s+decision)\b"
    ), "suicidal_ideation", "sudden_calm", Severity.ELEVATED),
    Rule("SI_MEANS_AVAILABLE", _compile(
        r"\b(?:i\s+have|i(?:'ve|\s+have)\s+got)\s+(?:the|my)\s+"
        r"(?:pills|weapon|gun|rope|knife)\s+(?:ready|with\s+me|beside\s+me|in\s+my\s+hand)\b"
    ), "suicidal_ideation", "means_available", Severity.URGENT, TemporalStatus.IMMINENT),
    Rule("SI_GENERIC_MEANS_WITH_INTENT", _compile(
        r"\b(?:kill\s+myself|end\s+my\s+life|take\s+my\s+own\s+life)"
        r"[^.!?]{0,100}\b(?:i\s+have\s+(?:the\s+)?means\s+with\s+me|"
        r"i\s+have\s+access\s+to\s+(?:the\s+)?means|the\s+means\s+are\s+with\s+me)\b|"
        r"\b(?:i\s+have\s+(?:the\s+)?means\s+with\s+me|"
        r"i\s+have\s+access\s+to\s+(?:the\s+)?means|the\s+means\s+are\s+with\s+me)"
        r"[^.!?]{0,100}\b(?:kill\s+myself|end\s+my\s+life|take\s+my\s+own\s+life)\b"
    ), "suicidal_ideation", "means_available", Severity.URGENT, TemporalStatus.IMMINENT),
    Rule("SELF_HARM_DIRECT", _compile(
        r"\bi\s+(?:cut|burn(?:ed|t)?|hurt|harm(?:ed)?|scratch(?:ed)?)\s+myself\b"
    ), "self_directed_harm", "direct_disclosure", Severity.ELEVATED),
    Rule("SELF_HARM_CURRENT", _compile(
        r"\bi(?:'m|\s+am)\s+(?:cutting|burning|hurting|harming|scratching)\s+myself\b"
    ), "self_directed_harm", "in_progress", Severity.URGENT, TemporalStatus.CURRENT),
    Rule("SELF_HARM_SLANG_RELIEF_LINES_CURRENT", _compile(
        r"\bi(?:'m|\s+am)\s+(?:making|adding|creating)\s+(?:new\s+)?relief\s+lines\b"
    ), "self_directed_harm", "contextual_slang_in_progress", Severity.URGENT,
         TemporalStatus.CURRENT, "contextual"),
    Rule("SELF_HARM_SLANG_RELIEF_LINES", _compile(
        r"\b(?:my|these)\s+(?:old\s+|new\s+)?relief\s+lines\b"
    ), "self_directed_harm", "contextual_slang_disclosure", Severity.ELEVATED,
         TemporalStatus.UNSPECIFIED, "contextual"),
    Rule("SELF_HARM_SLANG_BARCODE", _compile(
        r"\b(?:my|these)\s+(?:old\s+|new\s+)?barcodes?\b|"
        r"\bi(?:'m|\s+am)\s+barcod(?:ing|ed)\s+myself\b"
    ), "self_directed_harm", "contextual_slang_disclosure", Severity.ELEVATED,
         TemporalStatus.UNSPECIFIED, "contextual"),
    Rule("SELF_HARM_ABBREVIATION", _compile(
        r"\b(?:my\s+(?:s\s*/?\s*h|nssi)\s+(?:urges?|relapse)|"
        r"i\s+(?:relapsed|am\s+relapsing)\s+(?:with|into)\s+(?:s\s*/?\s*h|nssi))\b"
    ), "self_directed_harm", "contextual_abbreviation", Severity.ELEVATED,
         TemporalStatus.UNSPECIFIED, "contextual"),
    Rule("SEXUAL_VIOLENCE_CURRENT", _compile(
        r"\bi(?:'m|\s+am)\s+being\s+(?:raped|sexually\s+(?:assaulted|abused))\b|"
        r"\b(?:someone|he|she|they)\s+(?:is|are)\s+(?:raping|sexually\s+(?:assaulting|abusing))\s+me\b"
    ), "sexual_violence", "ongoing_disclosure", Severity.URGENT, TemporalStatus.CURRENT),
    Rule("SEXUAL_VIOLENCE_DISCLOSURE", _compile(
        r"\bi\s+(?:was|have\s+been)\s+(?:raped|sexually\s+(?:assaulted|abused|harassed))\b|"
        r"\b(?:someone|he|she|they)\s+(?:raped|sexually\s+(?:assaulted|abused))\s+me\b"
    ), "sexual_violence", "direct_disclosure", Severity.ELEVATED),
    Rule("DOMESTIC_VIOLENCE_CURRENT", _compile(
        r"\bmy\s+(?:partner|husband|wife|boyfriend|girlfriend|spouse|parent|father|mother)\s+"
        r"(?:is\s+)?(?:hitting|beating|strangling|choking|threatening)\s+me\b"
    ), "domestic_violence", "ongoing_disclosure", Severity.URGENT, TemporalStatus.CURRENT),
    Rule("DOMESTIC_VIOLENCE_DISCLOSURE", _compile(
        r"\bmy\s+(?:partner|husband|wife|boyfriend|girlfriend|spouse|parent|father|mother)\s+"
        r"(?:hit|beat|strangled|choked|threatened)\s+me\b"
    ), "domestic_violence", "direct_disclosure", Severity.ELEVATED),
    Rule("CHILD_ABUSE_CURRENT", _compile(
        r"\bi(?:'m|\s+am)\s+(?:a\s+(?:child|kid|minor)|under\s+18|1[0-7](?:\s+years?\s+old)?)"
        r"[^.!?]{0,100}\bmy\s+(?:parent|father|mother|guardian)\s+"
        r"(?:is\s+)?(?:hitting|beating|abusing|hurting)\s+me\b"
    ), "child_abuse_or_endangerment", "ongoing_disclosure", Severity.URGENT, TemporalStatus.CURRENT),
    Rule("CHILD_ABUSE_PAST", _compile(
        r"\bwhen\s+i\s+was\s+(?:a\s+)?(?:child|kid|young|younger)[^.!?]{0,80}\b(?:abused|molested|beaten|hurt)\b"
    ), "child_abuse_or_endangerment", "historical_disclosure", Severity.ELEVATED, TemporalStatus.PAST),
    Rule("ACUTE_PANIC_CURRENT", _compile(
        r"\bi(?:'m|\s+am)\s+having\s+(?:a\s+)?panic\s+attack(?:\s+right\s+now|\s+now)?\b"
    ), "acute_crisis", "acute_distress", Severity.ELEVATED, TemporalStatus.CURRENT),
    Rule("ACUTE_OVERDOSE_CURRENT", _compile(
        r"\bi\s+(?:just\s+)?(?:took|swallowed|drank)\s+(?:too\s+many|all\s+(?:of\s+)?my|a\s+large\s+amount\s+of)\b"
    ), "acute_crisis", "possible_medical_emergency", Severity.URGENT, TemporalStatus.CURRENT),
    Rule("HARM_OTHERS_INTENT", _compile(
        r"\bi\s+(?:want|intend|plan)\s+to\s+(?:kill|hurt|attack)\s+(?:him|her|them|someone|people)\b|"
        r"\bi(?:'m|\s+am)\s+going\s+to\s+(?:kill|hurt|attack)\s+(?:him|her|them|someone|people)\b"
    ), "harm_to_others", "explicit_intent", Severity.URGENT, TemporalStatus.IMMINENT),
)


TEMPORAL_PATTERNS = {
    TemporalStatus.CURRENT: _compile(
        r"\b(?:right\s+now|currently|at\s+this\s+moment|now|still|every\s+(?:day|night))\b"
    ),
    TemporalStatus.IMMINENT: _compile(r"\b(?:tonight|tomorrow|soon|later\s+today|in\s+(?:an?\s+)?hour)\b"),
    TemporalStatus.RECENT: _compile(r"\b(?:just|last\s+night|yesterday|this\s+(?:morning|week))\b"),
    TemporalStatus.PAST: _compile(
        r"\b(?:when\s+i\s+was\s+(?:a\s+)?(?:child|kid|young|younger)|years?\s+ago|"
        r"a\s+long\s+time\s+ago|back\s+then|in\s+(?:19\d{2}|20[01]\d))\b"
    ),
}

IDIOM_PATTERNS = {
    "IDIOM_DEAD_TIRED": _compile(r"\bdead\s+tired\b"),
    "IDIOM_DYING_LAUGHTER": _compile(r"\bdying\s+(?:of|from)\s+laughter\b"),
    "IDIOM_BODY_PART_HURTS": _compile(r"\b(?:back|head|eyes?|feet|legs?|arms?)\s+(?:is|are)?\s*killing\s+me\b"),
    "IDIOM_WORK_KILLING": _compile(r"\b(?:work|school|this\s+(?:meeting|assignment|exam))\s+is\s+killing\s+me\b"),
}

TOPIC_PATTERNS = {
    "grief": _compile(r"\b(?:grieving|bereavement|someone\s+i\s+love\s+died|lost\s+(?:my|someone))\b"),
    "substance_use": _compile(r"\b(?:withdrawal|relapsed|substance\s+use|addicted\s+to)\b"),
    "eating_disorder": _compile(r"\b(?:eating\s+disorder|anorexia|bulimia|binge\s+and\s+purge)\b"),
    "dissociation": _compile(r"\b(?:dissociat(?:e|ed|ing|ion)|depersonalization|derealization)\b"),
}

NEGATION = _compile(r"\b(?:not|never|don['’]?t|do\s+not|won['’]?t|will\s+not|wouldn['’]?t|no\s+longer)\b")
CONTRAST = _compile(r"\b(?:but|however|although|except)\b")
REPORTED_SPEECH = _compile(
    r"\b(?:he|she|they|my\s+(?:friend|partner|client|patient))\s+"
    r"(?:said|says|wrote|texted|told\s+me)\s*[,;:]?\s*[\"'“‘]?$"
)


def extract_rules(text: str) -> RuleExtraction:
    """Extract safety evidence from the normalized matching view."""

    idiom_spans = [
        (rule_id, match.start(), match.end())
        for rule_id, pattern in IDIOM_PATTERNS.items()
        for match in pattern.finditer(text)
    ]
    matches: list[SafetyMatch] = []
    for rule in SAFETY_RULES:
        for found in rule.pattern.finditer(text):
            # Explicit nearby time language wins over a rule's default. This
            # prevents "I had a plan and date picked years ago" from becoming
            # an imminent crisis merely because the plan rule is urgent.
            nearby_temporal = _temporal_near(text, found.start(), found.end())
            temporal = (
                nearby_temporal
                if nearby_temporal != TemporalStatus.UNSPECIFIED
                else rule.implied_temporal
            )
            suppression = (
                "local_negation"
                if rule.kind == "negated_disclosure"
                else _suppression_reason(text, found.start(), idiom_spans)
            )
            matches.append(SafetyMatch(
                rule_id=rule.rule_id,
                category=rule.category,
                kind=rule.kind,
                severity=rule.severity,
                temporal=temporal,
                span=Span(found.start(), found.end(), found.group(0)),
                suppressed=suppression is not None,
                suppression_reason=suppression,
                evidence_confidence=rule.confidence,
            ))

    # Preserve idiom recognition in the trace even when no safety rule matched.
    for rule_id, start, end in idiom_spans:
        matches.append(SafetyMatch(
            rule_id=rule_id,
            category=None,
            kind="idiom",
            severity=Severity.NONE,
            temporal=TemporalStatus.UNSPECIFIED,
            span=Span(start, end, text[start:end]),
            suppressed=True,
            suppression_reason="recognized_idiom",
        ))

    tags = sorted(tag for tag, pattern in TOPIC_PATTERNS.items() if pattern.search(text))
    matches.sort(key=lambda item: (item.span.start, item.span.end, item.rule_id))
    return RuleExtraction(matches=matches, topic_tags=tags, global_temporal=_global_temporal(text))


def _suppression_reason(text: str, start: int, idiom_spans: list[tuple[str, int, int]]) -> str | None:
    if any(idiom_start <= start < idiom_end for _, idiom_start, idiom_end in idiom_spans):
        return "recognized_idiom"
    if REPORTED_SPEECH.search(text[max(0, start - 120):start]):
        return "reported_speech"
    # Examine only the current clause. A contrast marker starts a new claim, so
    # "I am not going to hurt myself, but ..." cannot suppress evidence after "but".
    clause_start = max(text.rfind(mark, 0, start) for mark in (".", "!", "?", ";", ":", ",")) + 1
    prefix = text[clause_start:start]
    contrast_matches = list(CONTRAST.finditer(prefix))
    if contrast_matches:
        prefix = prefix[contrast_matches[-1].end():]
    words = re.findall(r"[\w'’]+", prefix)[-6:]
    if NEGATION.search(" ".join(words)):
        return "local_negation"
    return None


def _temporal_near(text: str, start: int, end: int) -> TemporalStatus:
    window = text[max(0, start - 90): min(len(text), end + 90)]
    statuses = [status for status, pattern in TEMPORAL_PATTERNS.items() if pattern.search(window)]
    return _combine_temporal(statuses)


def _global_temporal(text: str) -> TemporalStatus:
    statuses = [status for status, pattern in TEMPORAL_PATTERNS.items() if pattern.search(text)]
    return _combine_temporal(statuses)


def _combine_temporal(statuses: list[TemporalStatus]) -> TemporalStatus:
    unique = set(statuses)
    if not unique:
        return TemporalStatus.UNSPECIFIED
    if len(unique) > 1:
        return TemporalStatus.MIXED
    return next(iter(unique))
