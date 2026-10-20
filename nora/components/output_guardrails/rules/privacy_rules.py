from rules.base_rule import PatternRule

class PrivacyLeakRule(PatternRule):
    rule_id = 'PRIVACY-LEAK'
    reason_code = 'PRIVACY_LEAK'
    severity = 'CRITICAL'
    ignore_negated = False
    patterns = (r'\b(?:api[_ -]?key|access[_ -]?token|password|secret)\s*[:=]\s*[\w-]{8,}\b',
        r'\b(?:ssn|social security number)\s*[:=]?\s*\d{3}-\d{2}-\d{4}\b', r'\bsk-[a-z0-9_-]{20,}\b')
