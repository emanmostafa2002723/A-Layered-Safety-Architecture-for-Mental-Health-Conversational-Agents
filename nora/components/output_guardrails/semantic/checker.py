"""Fail-closed configurable model adapter implementing the frozen checker contract."""
from copy import deepcopy
from dataclasses import dataclass
import math
from time import monotonic
from baseline.contracts import checker_result, valid_policy_context, validate_checker_result
from config.fusion_policy import CRITICAL_CODES, effective_severity
from config.routes import required_behaviors, URGENT_ROUTES
from semantic.contract import (SEMANTIC_CODES, candidate_binding, initialize_semantic_contract,
    parse_assessment, ASSESSMENT_VERSION)
from semantic.strategy import build_messages, PROMPT_VERSION

@dataclass(frozen=True)
class SemanticConfig:
    model_id: str = 'unconfigured'
    model_revision: str = 'unconfigured'
    deployment_approved: bool = False
    timeout_seconds: float = 10.0

    def __post_init__(self):
        if any(type(v) is not str or not v.strip() or len(v) > 128
               for v in (self.model_id, self.model_revision)):
            raise ValueError('Invalid model identity')
        if type(self.deployment_approved) is not bool:
            raise ValueError('Invalid deployment approval')
        if (type(self.timeout_seconds) not in (float, int)
                or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0):
            raise ValueError('Invalid semantic timeout')
        if self.deployment_approved and 'unconfigured' in (self.model_id, self.model_revision):
            raise ValueError('Approved models require an explicit identity and revision')

class SemanticChecker:
    checker = 'semantic_safety'

    def __init__(self, transport=None, config=None):
        self.transport = transport
        self.config = SemanticConfig() if config is None else config
        if not isinstance(self.config, SemanticConfig):
            raise ValueError('Invalid semantic config')

    def _failure(self, error, status='ERROR', violations=None):
        return checker_result(self.checker, status, violations, {'error': error})

    def check(self, response, context):
        try:
            if (type(context) is not dict or set(context) != {'system_route', 'response_policy'}
                    or not valid_policy_context(response, context['system_route'], context['response_policy'])
                    or not response.strip() or len(response) > 20000):
                return self._failure('INVALID_SEMANTIC_INPUT', 'INVALID_INPUT')
            if self.transport is None:
                return self._failure('SEMANTIC_BACKEND_UNAVAILABLE')
            if not self.config.deployment_approved:
                return self._failure('SEMANTIC_MODEL_NOT_APPROVED')
            # Initialization is excluded from the model execution deadline.
            initialize_semantic_contract()
            snapshot = deepcopy(context)
            required = required_behaviors(snapshot['system_route'], snapshot['response_policy'])
            binding = candidate_binding(response, snapshot)
            messages = build_messages(response, snapshot, binding, required)
            started = monotonic()
            raw = self.transport.complete(messages, timeout_seconds=self.config.timeout_seconds)
            assessment = parse_assessment(raw, binding)
            findings = []
            uncertain = False
            for code in SEMANTIC_CODES:
                item = assessment['codes'][code]
                uncertain |= item['state'] == 'UNCERTAIN'
                if item['state'] == 'PRESENT':
                    findings.append({'reason_code': code,
                        'severity': effective_severity(code, 'HIGH', snapshot['system_route']),
                        'confidence': item['confidence'], 'metadata': {}})
            missing = []
            for behavior in sorted(required):
                state = assessment['behaviors'][behavior]['state']
                uncertain |= state == 'UNCERTAIN'
                if state == 'ABSENT':
                    missing.append(behavior)
            if missing:
                urgent = snapshot['system_route'] in URGENT_ROUTES
                code = 'EMERGENCY_UNDER_RESPONSE' if urgent else 'POLICY_MISMATCH'
                if not any(v['reason_code'] == code for v in findings):
                    findings.append({'reason_code': code,
                        'severity': effective_severity(code, 'HIGH', snapshot['system_route']),
                        'confidence': None, 'metadata': {}})
            if monotonic() - started > self.config.timeout_seconds:
                return self._failure('SEMANTIC_DEADLINE_EXCEEDED', 'TIMEOUT', findings)
            if uncertain:
                return self._failure('SEMANTIC_ASSESSMENT_UNCERTAIN', violations=findings)
            result = checker_result(self.checker, violations=findings, details={
                'assessment_version': ASSESSMENT_VERSION, 'prompt_version': PROMPT_VERSION,
                'model_id': self.config.model_id, 'model_revision': self.config.model_revision,
                'confidence_calibrated': False, 'missing_required_behaviors': missing,
                'requirements_satisfied': not missing})
            if not validate_checker_result(result, self.checker):
                return self._failure('INVALID_SEMANTIC_RESULT')
            return result
        except TimeoutError:
            return self._failure('SEMANTIC_BACKEND_TIMEOUT', 'TIMEOUT')
        except Exception:
            # No candidate, prompt, model text, credentials, or exception text in diagnostics.
            return self._failure('SEMANTIC_EXECUTION_FAILED')
