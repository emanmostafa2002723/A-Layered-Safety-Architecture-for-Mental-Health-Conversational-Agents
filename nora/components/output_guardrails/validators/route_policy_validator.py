from baseline.contracts import checker_result, violation, valid_policy_context
from baseline.deterministic_checker import DeterministicChecker
from config.routes import ROUTE_REQUIREMENTS, URGENT_ROUTES
from config.routes import route_requirements_missing, clinical_boundary_policy_weakened
from config.taxonomy import CLINICAL_CODES

class RoutePolicyValidator:
    checker = 'route_policy_validator'

    def __init__(self, approved_templates=None):
        # Trusted integration configuration, never supplied by candidate/evidence.
        self.approved_templates = dict(approved_templates or {})

    def validate(self, generated_response, system_route, response_policy, safety_violations=None):
        try:
            if not valid_policy_context(generated_response, system_route, response_policy):
                return checker_result(self.checker, 'INVALID_INPUT')
            if safety_violations is None:
                safety = DeterministicChecker().check(generated_response, {'system_route': system_route})
                if safety['status'] != 'OK':
                    return checker_result(self.checker, safety['status'])
                safety_violations = safety['violations']
            codes = {v['reason_code'] for v in safety_violations}
            mismatches = []
            if route_requirements_missing(system_route, response_policy):
                mismatches.append('ROUTE_REQUIREMENTS_MISSING_FROM_POLICY')
            if codes & set(response_policy['prohibited_behaviors']):
                mismatches.append('PROHIBITED_BEHAVIOR_PRESENT')
            if response_policy['personalized_conclusions_prohibited'] and codes & CLINICAL_CODES:
                mismatches.append('PERSONALIZED_CONCLUSION_PRESENT')
            if clinical_boundary_policy_weakened(system_route, response_policy):
                mismatches.append('CLINICAL_BOUNDARY_POLICY_WEAKENED')
            if not response_policy['may_continue_with_general_information']:
                approved = self.approved_templates.get(response_policy['template_id'])
                if approved is None:
                    return checker_result(self.checker, 'ERROR', details={'error': 'APPROVED_TEMPLATE_UNAVAILABLE'})
                if generated_response != approved:
                    mismatches.append('TEMPLATE_ONLY_POLICY_MISMATCH')
            findings = [violation('POLICY_MISMATCH', 'CRITICAL' if system_route in URGENT_ROUTES else 'HIGH')] if mismatches else []
            return checker_result(self.checker, violations=findings, details={'mismatches': mismatches})
        except Exception:
            return checker_result(self.checker, 'ERROR')
