import re
import unicodedata
from abc import ABC, abstractmethod
from baseline.contracts import violation

def _canonical_text(text):
    return unicodedata.normalize('NFKC', text).casefold().replace('\u2019', "'")

def normalize(text):
    return re.sub(r'\s+', ' ', _canonical_text(text))

def clauses(text):
    # Split before whitespace collapse: a negation must not cross a line boundary.
    parts = re.split(r'(?<!\d)\.(?!\d)|[!?;\r\n\u2028\u2029]+|\b(?:but|however|then)\b', _canonical_text(text))
    return [re.sub(r'\s+', ' ', part) for part in parts]

def negated_prefix(clause, start):
    prefix = clause[max(0, start - 48):start]
    return bool(re.search(r"\b(?:do not|don't|never|not|cannot|can't|avoid)\s+(?:\w+\s+){0,2}$", prefix))

class DeterministicRule(ABC):
    @abstractmethod
    def check(self, response, context):
        raise NotImplementedError

class PatternRule(DeterministicRule):
    rule_id = ''
    reason_code = ''
    severity = 'HIGH'
    patterns = ()
    ignore_negated = True

    def check(self, response, context):
        for clause in clauses(response):
            for pattern in self.patterns:
                for match in re.finditer(pattern, clause):
                    if not self.ignore_negated or not negated_prefix(clause, match.start()):
                        return [violation(self.reason_code, self.severity, self.rule_id)]
        return []
