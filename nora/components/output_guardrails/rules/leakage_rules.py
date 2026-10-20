from rules.base_rule import PatternRule

class SystemPromptLeakRule(PatternRule):
    rule_id = 'SYSTEM-PROMPT-LEAK'
    reason_code = 'SYSTEM_PROMPT_LEAKAGE'
    ignore_negated = False
    patterns = (r'\b(?:my|the|hidden) system (?:prompt|instructions?)\s*(?:is|are|says|:)\s*', r'\b(?:begin|end) (?:system|developer) (?:prompt|instructions?)\b', r'<\|(?:system|developer)\|>')
