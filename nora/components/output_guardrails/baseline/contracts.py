"""Offline JSON Schema validation. Remote schema retrieval is never enabled."""
import json
import math
from functools import lru_cache
from pathlib import Path
from config.severity import Severity
SCHEMA_ROOT = Path(__file__).resolve().parents[1] / 'schemas'
SCHEMA_FILES = ('input_schema.json', 'response_policy_schema.json', 'evidence_schema.json',
    'finding_schema.json', 'checker_result_schema.json', 'decision_schema.json')

def checker_result(checker, status='OK', violations=None, details=None):
    return {'checker': checker, 'status': status,
        'violations': [] if violations is None else violations,
        'details': {} if details is None else details}

def violation(code, severity=Severity.HIGH, rule_id=None):
    return {'reason_code': code, 'severity': severity, 'confidence': None,
        'metadata': {} if rule_id is None else {'rule_id': rule_id}}

def _json_safe(value, depth=0):
    if depth > 12:
        return False
    if value is None or type(value) in (str, bool, int):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return len(value) <= 256 and all(_json_safe(v, depth + 1) for v in value)
    if type(value) is dict:
        return len(value) <= 64 and all(type(k) is str and _json_safe(v, depth + 1) for k, v in value.items())
    return False

@lru_cache(maxsize=1)
def _validators():
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    documents = {name: json.loads((SCHEMA_ROOT / name).read_text(encoding='utf-8')) for name in SCHEMA_FILES}
    registry = Registry().with_resources((s['$id'], Resource.from_contents(s)) for s in documents.values())
    validators = {}
    for name, schema in documents.items():
        Draft202012Validator.check_schema(schema)
        validators[name] = Draft202012Validator(schema, registry=registry)
    return validators

def validate_schema(name, value):
    return _json_safe(value) and _validators()[name].is_valid(value)

def initialize_contracts():
    """Load/check local schemas before starting a per-candidate deadline."""
    _validators()

def parse_json_object(raw, max_bytes):
    """Strict bounded model JSON: reject duplicates, non-finite numbers and extras later."""
    if type(raw) is not str or len(raw.encode('utf-8')) > max_bytes:
        raise ValueError('Invalid JSON payload')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON field')
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=unique)
    if type(value) is not dict or not _json_safe(value):
        raise ValueError('Invalid JSON object')
    return value

def validate_checker_result(value, expected_checker):
    return validate_schema('checker_result_schema.json', value) and value['checker'] == expected_checker

def valid_individual_findings(value, expected_checker=None):
    """Salvage only individually validated findings, with a known envelope identity."""
    from copy import deepcopy
    if type(value) is not dict:
        return []
    name = value.get('checker')
    if type(name) is not str or not 0 < len(name) <= 256 or expected_checker is not None and name != expected_checker:
        return []
    candidates = value.get('violations')
    if type(candidates) is not list or len(candidates) > 128:
        return []
    return [deepcopy(f) for f in candidates if validate_schema('finding_schema.json', f)]

def input_contract_error(value):
    """Shared structural checks run before copying or dispatching caller data."""
    if not validate_schema('input_schema.json', value):
        return 'INPUT_SCHEMA'
    if not value['generated_response'].strip():
        return 'EMPTY_RESPONSE'
    if type(value['execution_context']['rewrite_attempt']) is not int:
        return 'REWRITE_ATTEMPT_TYPE'
    evidence = value['verification_evidence']
    if len({item['evidence_id'] for item in evidence}) != len(evidence):
        return 'DUPLICATE_EVIDENCE_ID'
    return None

def valid_final_result(result, context, config=None, findings=()):
    """Structural schema plus action authorization in its authoritative context."""
    from config.fusion_policy import canonical_reasons, policy_floor_missing, rewrite_eligible
    try:
        if not validate_schema('decision_schema.json', result):
            return False
        if result['reason_codes'] != canonical_reasons(result['reason_codes']):
            return False
        if input_contract_error(context) is not None:
            return (result['guardrail_status'] != 'OK'
                and (result['decision'], result['action'], result['final_response']) == ('BLOCK', 'ABSTAIN', None))
        if result['decision'] == 'ALLOW':
            return bool(not policy_floor_missing(context) and result['final_response'].strip()
                and result['final_response'] == context['generated_response'])
        if result['action'] == 'REWRITE':
            return rewrite_eligible(context, result['reason_codes'], config, findings)
        if result['action'] == 'SAFE_FALLBACK':
            return config is not None and context['system_route'] in config.approved_fallback_routes
        return result['action'] == 'ABSTAIN'
    except Exception:
        return False

def valid_policy_context(response, route, policy):
    """Shared structural precondition for standalone policy/safety validators."""
    from config.routes import SUPPORTED_ROUTES
    return (type(route) is str and route in SUPPORTED_ROUTES
        and isinstance(response, str) and validate_schema('response_policy_schema.json', policy))
