"""Small, explicit user-report notes; never a diagnosis or a safety state."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re
import math


REPORTS = {
    ('sleep', 'reported'): 'reported difficulty sleeping',
    ('sleep', 'improved'): 'reported that sleep had improved',
    ('stress', 'reported'): 'reported feeling stressed or overwhelmed',
    ('stress', 'improved'): 'reported feeling less stressed',
    ('support_preference', 'listening'): 'preferred listening rather than advice',
}


@dataclass(frozen=True, slots=True)
class MemoryNote:
    topic: str
    state: str
    observed_at: float
    expires_at: float

    def __post_init__(self):
        if ((self.topic, self.state) not in REPORTS
                or not all(isinstance(t,(int,float)) and math.isfinite(t)
                           for t in (self.observed_at,self.expires_at))
                or not 0 <= self.observed_at < self.expires_at):
            raise ValueError('invalid memory note')

    def prompt_dict(self):
        return {'user_report': REPORTS[self.topic, self.state],
                'reported_on_utc': datetime.fromtimestamp(self.observed_at, timezone.utc).date().isoformat()}


_PATTERNS = (
    ('sleep', 'reported', r"i (?:can'?t|cannot|can not) sleep(?: well)?(?: lately| recently| at night)?"),
    ('sleep', 'reported', r"i (?:am|'m|have been) (?:having trouble sleeping|sleeping poorly)(?: lately| recently)?"),
    ('sleep', 'reported', r"i (?:have|had) trouble sleeping(?: lately| recently)?"),
    ('sleep', 'improved', r"i (?:am|'m|have been) sleeping (?:better|well)(?: now| lately)?"),
    ('stress', 'reported', r"i (?:am|'m|feel|have been feeling) (?:very |really |so )?(?:stressed|overwhelmed|under stress)(?: lately| recently| today)?"),
    ('stress', 'improved', r"i (?:am|'m|feel) (?:less stressed|not stressed anymore)(?: now)?"),
    ('support_preference', 'listening', r"i (?:prefer|would prefer) (?:you (?:to )?)?(?:just )?(?:listening|to listen) (?:rather than|instead of) (?:giving )?advice"),
)


def explicit_reports(text: str) -> tuple[tuple[str, str], ...]:
    """Deliberately low-recall: whole first-person sentences, no free-text notes.

    Quoted, conditional, third-party, diagnostic, medication and crisis content
    is not extracted. No assistant output is ever passed to this function.
    """
    text = text.casefold().replace('\u2019', "'").strip()
    text = re.sub(r"\bi'm\b", 'i am', text)
    if len(text) > 2000 or re.search(r'["“”]|\b(?:if|imagine|suppose|quote|fiction\w*|example|pretend|joking|role.play|suicid\w*|kill|self.harm|diagnos\w*|medicat\w*|do not remember|don\x27t remember|forget)\b', text):
        return ()
    found = {}
    for part in re.finditer(r'([^.!?\n]+)([.!?\n]|$)', text):
        if part.group(2) == '?':
            continue  # A first-person question is not a declarative report.
        sentence = ' '.join(part.group(1).split())
        sentence = re.sub(r'^actually[, ]+', '', sentence)
        for topic, state, pattern in _PATTERNS:
            if re.fullmatch(pattern, sentence):
                found[topic] = state
    if re.search(r'\b(?:actually|not true|not anymore|no longer)\b', text):
        # Ambiguous retractions must not leave a new negative-health assertion.
        found = {topic:state for topic,state in found.items() if state != 'reported'}
    return tuple(found.items())


def select_notes(notes, text: str, *, now: float) -> tuple[MemoryNote, ...]:
    """No unsolicited memory on 'hi'; permit relevant check-ins when invited."""
    text = text.casefold()
    broad = bool(re.search(r'\b(?:remember|last time|check.in|how have i|what do you know about me)\b', text))
    terms = {'sleep': r'\b(?:sleep\w*|tired|insomnia|night)\b',
             'stress': r'\b(?:stress\w*|overwhelmed|pressure)\b',
             'support_preference': r'\b(?:listen\w*|advice|support|feel\w*)\b'}
    targeted = {topic for topic,pattern in terms.items() if re.search(pattern,text)}
    general_recall = broad and (not re.search(r'\b(?:about|my)\b',text)
                               or bool(re.search(r'what do you know about me',text)))
    return tuple(n for n in notes if n.observed_at <= now < n.expires_at
                 and (n.topic in targeted if targeted else general_recall))[:3]


def memory_recall_request(text: str) -> bool:
    return bool(re.search(r'\b(?:(?:do|can) you remember|what did i (?:say|tell)|'
                          r'last (?:time|chat)|previous (?:chat|conversation)|'
                          r'remember (?:me|what i|my sleep|our)|what do you know about me)\b', text, re.I))


def factual_recall_fallback(notes: tuple[MemoryNote, ...]) -> str:
    """Exact database-backed recall, not invented advice; still requires output approval."""
    if not notes:
        return ('I do not have those earlier details available here. '
                'Would you like to tell me what you want to pick up from?')
    parts = []
    for note in notes:
        data = note.prompt_dict()
        parts.append('Your saved note from '+data['reported_on_utc']+' says you '+data['user_report']+'.')
    return ' '.join(parts)+' Has anything changed since then?'


def exact_recall_request(text: str) -> bool:
    """A database-recall question only, not a mixed support/clinical request."""
    return bool(re.fullmatch(r'\s*(?:(?:do|can) you remember (?:me|what i (?:said|told you)'
        r'(?: about (?:my )?(?:sleep|stress|listening preference))?)|'
        r'what did i (?:say|tell you) about (?:my )?(?:sleep|stress)|what do you know about me)\s*[?.!]*\s*', text, re.I))
