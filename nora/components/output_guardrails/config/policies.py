"""Step 5B entry points. No retrieval, generation, route selection, or display."""
from copy import deepcopy
from dataclasses import replace
from time import monotonic
import math
from baseline.contracts import (checker_result, validate_checker_result, valid_final_result,
    valid_individual_findings, initialize_contracts, input_contract_error)
from baseline.execution import BoundedChecker, DeadlineChecker
from baseline.worker_spec import as_worker_spec
from config.fusion_policy import execution_health
from baseline.decision_mapper import DecisionMapper
from baseline.deterministic_checker import DeterministicChecker
from config.runtime import RuntimeConfig
from validators.contract_validator import ContractValidator
from validators.route_policy_validator import RoutePolicyValidator
from validators.required_safety_validator import RequiredSafetyValidator

def _execute(checker, operation, timeout_seconds):
    """Evaluation-only adapter; production supplies only bounded external checkers."""
    started = monotonic()
    try:
        result = operation()
        if not validate_checker_result(result, checker):
            return checker_result(checker, 'ERROR', valid_individual_findings(result, checker),
                details={'error': 'MALFORMED_CHECKER_RESULT'})
        if monotonic() - started > timeout_seconds:
            return checker_result(checker, execution_health([result['status'], 'TIMEOUT']), result['violations'])
        return result
    except TimeoutError:
        return checker_result(checker, 'TIMEOUT')
    except Exception:
        return checker_result(checker, 'ERROR', details={'error': 'CHECKER_EXECUTION_FAILED'})

def evaluate_deterministic_baseline(guardrail_input, config=None, *, deterministic_checker=None,
        evidence_checker=None, approved_templates=None):
    """Returns (frozen result, checker diagnostics). Evaluation-only baseline."""
    runtime = RuntimeConfig() if config is None else config
    mapper = DecisionMapper(runtime, evaluation_only=True)
    # Dependency/schema startup is distinct from processing a candidate.
    try:
        initialize_contracts()
    except Exception:
        failed = checker_result('contract_validator', 'ERROR', details={'error': 'CONTRACT_INITIALIZATION_FAILED'})
        return mapper.failure(guardrail_input), [failed]
    contract = _execute('contract_validator', lambda: ContractValidator().validate(guardrail_input), runtime.checker_timeout_seconds)
    if contract['status'] != 'OK':
        return mapper.map([contract], guardrail_input), [contract]
    context = deepcopy(guardrail_input)
    response, route, policy = context['generated_response'], context['system_route'], context['response_policy']
    safety_checker = DeterministicChecker(timeout_seconds=runtime.checker_timeout_seconds) if deterministic_checker is None else deterministic_checker
    safety = _execute('deterministic_safety', lambda: safety_checker.check(response, deepcopy(context)), runtime.checker_timeout_seconds)
    route_validator = RoutePolicyValidator(approved_templates)
    routing = _execute('route_policy_validator', lambda: route_validator.validate(response, route, deepcopy(policy), deepcopy(safety['violations'])), runtime.checker_timeout_seconds)
    required = _execute('required_safety_checker', lambda: RequiredSafetyValidator().validate(response, route, deepcopy(policy)), runtime.checker_timeout_seconds)
    if not policy['factual_verification']['required']:
        evidence = checker_result('evidence_checker', 'NOT_APPLICABLE', details={'applicability': 'NOT_REQUIRED_BY_POLICY'})
    elif evidence_checker is None:
        # Presence/word overlap does not prove support. Step 5B has no factual verifier.
        evidence = checker_result('evidence_checker', 'ERROR', details={'error': 'FACTUAL_VERIFIER_UNAVAILABLE'})
    else:
        approved = [deepcopy(e) for e in context['verification_evidence'] if e['review_status'] == 'approved']
        evidence = _execute('evidence_checker', lambda: evidence_checker.check(response, approved), runtime.evidence_checker_timeout_seconds)
    checks = [contract, safety, routing, required, evidence]
    result = mapper.map(checks, context)
    try:
        if not valid_final_result(result, context, runtime):
            result = mapper.failure(context)
    except Exception:
        result = mapper.failure(context)
    return result, checks

def run_deterministic_baseline(guardrail_input, config=None, **kwargs):
    """For Step 5B evaluation/tests only; not the final display authorization gate."""
    try:
        return evaluate_deterministic_baseline(guardrail_input, config, **kwargs)[0]
    except Exception:
        return DecisionMapper().failure(guardrail_input)

def _evaluate_output_guardrail(guardrail_input, config=None, *, semantic_checker=None, **kwargs):
    """Internal decision evaluation. Protocol tests may call this without production I/O."""
    try:
        runtime = RuntimeConfig() if config is None else config
        mapper = DecisionMapper(runtime)
        # Keep the full decision bound to one snapshot, even if callbacks mutate caller data.
        context = deepcopy(guardrail_input)
        result, checks = evaluate_deterministic_baseline(context, runtime, **kwargs)
        if checks[0]['status'] != 'OK':
            return result
        if semantic_checker is None:
            semantic = checker_result('semantic_safety', 'ERROR', details={'error': 'SEMANTIC_CHECKER_UNAVAILABLE'})
        else:
            semantic = _execute('semantic_safety', lambda: semantic_checker.check(
                context['generated_response'], {
                    'system_route': context['system_route'],
                    'response_policy': deepcopy(context['response_policy']),
                }), runtime.semantic_checker_timeout_seconds)
            if semantic['status'] == 'NOT_APPLICABLE':
                semantic = checker_result('semantic_safety', 'ERROR')
        final = mapper.map(checks + [semantic], context,
            required_checkers=tuple(c['checker'] for c in checks) + ('semantic_safety',))
        return final if valid_final_result(final, context, runtime) else mapper.failure(context)
    except Exception:
        return DecisionMapper().failure(guardrail_input)

def run_output_guardrail(guardrail_input, config=None, *, semantic_checker=None,
        evidence_checker=None, deterministic_checker=None, approved_templates=None,
        rewriter=None, fallback_registry=None, deadline=None):
    """Bounded production decision gate. Returns an action request, never executes it."""
    started = monotonic()
    try:
        initialize_contracts()
        runtime = RuntimeConfig() if config is None else config
        if type(runtime) is not RuntimeConfig:
            raise ValueError('Invalid runtime')
        if deadline is not None and (type(deadline) not in (float, int) or not math.isfinite(deadline)):
            raise ValueError('Invalid request deadline')
        deadline = min(started + runtime.request_timeout_seconds,
            deadline if deadline is not None else started + runtime.request_timeout_seconds)
        mapper = DecisionMapper(runtime)
        # Reject non-JSON objects before deepcopy can invoke caller hooks.
        if input_contract_error(guardrail_input) is not None:
            return mapper.failure(guardrail_input, 'INVALID_INPUT')
        context = deepcopy(guardrail_input)
        from delivery.fallbacks import FallbackRegistry
        registry = FallbackRegistry() if fallback_registry is None else fallback_registry
        if type(registry) is not FallbackRegistry:
            return {**mapper.failure(context), 'action': 'ABSTAIN'}
        rewrite_available = False
        if runtime.controlled_rewrite_available and rewriter is not None:
            try:
                as_worker_spec(rewriter)
                rewrite_available = True
            except Exception:
                pass
        runtime = replace(runtime, controlled_rewrite_available=rewrite_available,
            approved_fallback_routes=registry.available_routes)
        mapper = DecisionMapper(runtime)
        if (approved_templates is not None and (type(approved_templates) is not dict
                or len(approved_templates) > 64 or any(type(k) is not str or type(v) is not str
                or not k.strip() or len(k) > 256 or not v.strip() or len(v) > 20000
                for k, v in approved_templates.items()))):
            failed = mapper.failure(context)
            if failed['action'] == 'SAFE_FALLBACK' and registry.resolve(context['system_route'], context['response_policy']) is None:
                failed['action'] = 'ABSTAIN'
            return failed
        def bounded(target, name, timeout):
            if target is None:
                return None
            if type(target) is BoundedChecker:
                return DeadlineChecker(target.target, name, min(timeout, target.timeout_seconds), deadline)
            return DeadlineChecker(target, name, timeout, deadline)
        result = _evaluate_output_guardrail(context, runtime,
            semantic_checker=bounded(semantic_checker, 'semantic_safety', runtime.semantic_checker_timeout_seconds),
            evidence_checker=bounded(evidence_checker, 'evidence_checker', runtime.evidence_checker_timeout_seconds),
            deterministic_checker=bounded(deterministic_checker, 'deterministic_safety', runtime.checker_timeout_seconds),
            approved_templates=deepcopy(approved_templates))
        if result['action'] == 'SAFE_FALLBACK' and registry.resolve(context['system_route'], context['response_policy']) is None:
            result = {**result, 'action': 'ABSTAIN'}
        if monotonic() > deadline:
            result = mapper.failure(context, execution_health([result['guardrail_status'], 'TIMEOUT']), result['reason_codes'])
        return result if valid_final_result(result, context, runtime) else mapper.failure(context)
    except Exception:
        return DecisionMapper().failure(guardrail_input)
