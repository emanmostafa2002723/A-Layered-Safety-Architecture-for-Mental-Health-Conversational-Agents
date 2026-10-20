"""Internal model protocol. Does not change any Step 5A public schema."""
import hashlib
import json
from functools import lru_cache
from baseline.contracts import parse_json_object
from config.taxonomy import REASON_CODES, REQUIRED_BEHAVIORS

ASSESSMENT_VERSION = 'semantic-assessment-1.0'
# Factual support is assessed independently against governed RAG evidence.
SEMANTIC_CODES = tuple(c for c in REASON_CODES if c != 'UNSUPPORTED_MEDICAL_CLAIM')
ASSESSMENT_STATES = ('PRESENT', 'ABSENT', 'UNCERTAIN')
MAX_OUTPUT_BYTES = 32768

def output_schema():
    assessment = {'type': 'object', 'required': ['state', 'confidence'],
        'properties': {'state': {'enum': list(ASSESSMENT_STATES)},
            'confidence': {'type': ['number', 'null'], 'minimum': 0, 'maximum': 1}},
        'additionalProperties': False}
    def coverage(keys):
        return {'type': 'object', 'required': list(keys),
            'properties': {key: {'$ref': '#/$defs/assessment'} for key in keys},
            'additionalProperties': False}
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema',
        'type': 'object', 'required': ['schema_version', 'candidate_binding', 'codes', 'behaviors'],
        'properties': {'schema_version': {'const': ASSESSMENT_VERSION},
            'candidate_binding': {'type': 'string', 'pattern': '^[a-f0-9]{64}$'},
            'codes': coverage(SEMANTIC_CODES), 'behaviors': coverage(REQUIRED_BEHAVIORS)},
        '$defs': {'assessment': assessment}, 'additionalProperties': False}

@lru_cache(maxsize=1)
def initialize_semantic_contract():
    from jsonschema import Draft202012Validator
    schema = output_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)

def candidate_binding(response, context):
    payload = {'generated_response': response, **context, 'schema_version': ASSESSMENT_VERSION}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()

def parse_assessment(raw, binding):
    value = parse_json_object(raw, MAX_OUTPUT_BYTES)
    if (not initialize_semantic_contract().is_valid(value)
            or value['candidate_binding'] != binding):
        raise ValueError('Invalid model assessment')
    return value
