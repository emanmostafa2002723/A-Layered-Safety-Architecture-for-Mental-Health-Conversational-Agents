from baseline.contracts import input_contract_error, valid_final_result, violation
from baseline.findings import collect_findings
from config.routes import required_behaviors
from config.runtime import RuntimeConfig
from config.fusion_policy import (CORE_CHECKERS, PRODUCTION_CHECKERS, CRITICAL_CODES,
    REWRITEABLE_CODES, canonical_reasons, policy_floor_missing, rewrite_eligible)

class DecisionMapper:
    def __init__(self, config=None, *, evaluation_only=False):
        self.config = RuntimeConfig() if config is None else config
        if type(self.config) is not RuntimeConfig or type(evaluation_only) is not bool:
            raise ValueError('Invalid fusion configuration')
        self.evaluation_only = evaluation_only

    def fallback_action(self, context):
        try:
            if input_contract_error(context) is None and context['system_route'] in self.config.approved_fallback_routes:
                return 'SAFE_FALLBACK'
        except Exception:
            pass
        return 'ABSTAIN'

    def failure(self, context, status='ERROR', reason_codes=None):
        return {'schema_version': 'output-guardrail-result-1.0', 'guardrail_status': status,
            'decision': 'BLOCK', 'reason_codes': canonical_reasons(reason_codes or []),
            'action': self.fallback_action(context), 'final_response': None}

    def map(self, checker_results, context, required_checkers=()):
        try:
            route = context.get('system_route') if type(context) is dict and type(context.get('system_route')) is str else None
            collected = collect_findings(checker_results, route)
            by_name, findings = collected.by_name, list(collected.violations)
            reasons = list(collected.reason_codes)
            if input_contract_error(context) is not None:
                return self.failure(context, 'INVALID_INPUT', reasons)
            if collected.health_status != 'OK':
                return self.failure(context, collected.health_status, reasons)
            mandatory = set(CORE_CHECKERS if self.evaluation_only else PRODUCTION_CHECKERS) | set(required_checkers)
            if not mandatory <= set(by_name):
                return self.failure(context, reason_codes=reasons)
            optional = {'required_safety_checker', 'evidence_checker'}
            if any(by_name[name]['status'] != 'OK' for name in mandatory - optional):
                return self.failure(context, reason_codes=reasons)
            if context['response_policy']['factual_verification']['required'] and by_name['evidence_checker']['status'] != 'OK':
                return self.failure(context, reason_codes=reasons)
            required = required_behaviors(route, context['response_policy'])
            if required and by_name['required_safety_checker']['status'] != 'OK':
                return self.failure(context, reason_codes=reasons)
            if required:
                details = by_name['required_safety_checker']['details']
                missing = details.get('missing_required_behaviors')
                if (type(details.get('requirements_satisfied')) is not bool or type(missing) is not list
                        or any(type(v) is not str or v not in required for v in missing)
                        or len(set(missing)) != len(missing)
                        or details['requirements_satisfied'] != (not missing)
                        or missing and not by_name['required_safety_checker']['violations']):
                    return self.failure(context, reason_codes=reasons)
            if policy_floor_missing(context):
                findings.append(violation('POLICY_MISMATCH', 'CRITICAL' if route in {'crisis_response', 'urgent_medical_response'} else 'HIGH'))
                reasons = canonical_reasons(reasons + ['POLICY_MISMATCH'])
            if not findings:
                result = {'schema_version': 'output-guardrail-result-1.0', 'guardrail_status': 'OK',
                    'decision': 'ALLOW', 'reason_codes': [], 'action': 'DISPLAY',
                    'final_response': context['generated_response']}
            else:
                result = self.failure(context, 'OK', reasons)
                if rewrite_eligible(context, reasons, self.config, findings):
                    result['action'] = 'REWRITE'
            return result if valid_final_result(result, context, self.config, findings) else self.failure(context, reason_codes=reasons)
        except Exception:
            return self.failure(context)
