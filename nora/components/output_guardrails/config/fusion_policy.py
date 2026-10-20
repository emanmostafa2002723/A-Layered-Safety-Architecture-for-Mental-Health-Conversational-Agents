"""Approved Step 8 policy; public Step 5A vocabularies remain unchanged."""
from config.routes import URGENT_ROUTES, ROUTE_REQUIREMENTS
from config.routes import route_requirements_missing, clinical_boundary_policy_weakened
from config.taxonomy import REASON_CODES

POLICY_VERSION = 'fusion-8-1.0'
CORE_CHECKERS = ('contract_validator', 'deterministic_safety', 'route_policy_validator',
    'required_safety_checker', 'evidence_checker')
PRODUCTION_CHECKERS = CORE_CHECKERS + ('semantic_safety',)
REWRITEABLE_CODES = frozenset({'POLICY_MISMATCH', 'UNSUPPORTED_MEDICAL_CLAIM'})
REWRITE_ROUTES = frozenset({'normal_chat', 'monitored_support'})
CRITICAL_CODES = frozenset({'PERSONALIZED_MEDICATION_INSTRUCTION', 'HARMFUL_INSTRUCTIONS',
    'EMERGENCY_GUIDANCE_CONTRADICTION', 'PRIVACY_LEAK', 'SYSTEM_PROMPT_LEAKAGE'})
SEVERITY_RANK = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2, 'CRITICAL': 3}
HEALTH_PRIORITY = ('INVALID_INPUT', 'TIMEOUT', 'ERROR', 'OK')

def effective_severity(code, declared, route=None):
    floor = 'CRITICAL' if code in CRITICAL_CODES or (
        code == 'EMERGENCY_UNDER_RESPONSE' and route in URGENT_ROUTES) else 'HIGH'
    return max((declared, floor), key=SEVERITY_RANK.__getitem__)

def canonical_reasons(codes):
    observed = set(codes)
    return [code for code in REASON_CODES if code in observed]

def execution_health(statuses):
    return next((state for state in HEALTH_PRIORITY if state in statuses), 'OK')

def policy_floor_missing(context):
    route, policy = context['system_route'], context['response_policy']
    return (route_requirements_missing(route, policy)
        or clinical_boundary_policy_weakened(route, policy))

def rewrite_eligible(context, reasons, config=None, findings=()):
    return (bool(reasons) and context['system_route'] in REWRITE_ROUTES
        and set(reasons) <= REWRITEABLE_CODES and not policy_floor_missing(context)
        and not any(effective_severity(v['reason_code'], v['severity'], context['system_route']) == 'CRITICAL'
            for v in findings)
        and (config is None or config.controlled_rewrite_available
            and context['execution_context']['rewrite_attempt'] < config.max_rewrite_attempts))
