from time import monotonic
from baseline.contracts import checker_result, validate_schema

class DeterministicChecker:
    checker = 'deterministic_safety'

    def __init__(self, rules=None, timeout_seconds=0.5):
        if rules is None:
            from rules.registry import RULE_REGISTRY
            rules = RULE_REGISTRY
        self.rules = tuple(rules)
        self.timeout_seconds = timeout_seconds

    def check(self, response, context):
        findings = []
        started = monotonic()
        try:
            if not self.rules:
                return checker_result(self.checker, 'ERROR', details={'error': 'EMPTY_RULE_SET'})
            for rule in self.rules:
                found = rule.check(response, context)
                if monotonic() - started > self.timeout_seconds:
                    return checker_result(self.checker, 'TIMEOUT', findings)
                if type(found) is not list or not all(validate_schema('finding_schema.json', f) for f in found):
                    return checker_result(self.checker, 'ERROR', findings, {'error': 'MALFORMED_RULE_RESULT'})
                findings.extend(found)
            return checker_result(self.checker, violations=findings)
        except TimeoutError:
            return checker_result(self.checker, 'TIMEOUT', findings)
        except Exception:
            return checker_result(self.checker, 'ERROR', findings, {'error': 'RULE_EXECUTION_FAILED'})
