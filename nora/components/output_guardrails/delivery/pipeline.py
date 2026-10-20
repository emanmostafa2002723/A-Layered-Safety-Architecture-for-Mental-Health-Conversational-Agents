"""Step 7 delivery entry point. Executes actions without changing guardrail schemas."""
from copy import deepcopy
from dataclasses import dataclass, replace
from time import monotonic
from baseline.contracts import valid_final_result
from baseline.decision_mapper import DecisionMapper
from baseline.execution import BoundedChecker, ExecutionFailure, bounded_invoke
from baseline.recheck import prepare_rewrite_input
from config.policies import run_output_guardrail
from config.runtime import RuntimeConfig
from delivery.fallbacks import FallbackRegistry
from validators.contract_validator import ContractValidator
from config.routes import SUPPORTED_ROUTES

@dataclass(frozen=True)
class DeliveryResult:
    guardrail_result: dict
    delivery_action: str
    delivered_text: object
    route: object
    template_id: object = None
    template_version: object = None
    evaluations: int = 0

def _blocked_delivery(context, result, registry, evaluations):
    route = context.get('system_route') if type(context) is dict else None
    if type(route) is not str or route not in SUPPORTED_ROUTES:
        route = None
    if result['action'] == 'SAFE_FALLBACK' and ContractValidator().validate(context)['status'] == 'OK':
        template = registry.resolve(route, context['response_policy'])
        if template is not None:
            return DeliveryResult(deepcopy(result), 'SAFE_FALLBACK', template.text, route,
                template.template_id, template.version, evaluations)
    blocked = deepcopy(result)
    blocked.update(decision='BLOCK', action='ABSTAIN', final_response=None)
    return DeliveryResult(blocked, 'ABSTAIN', None, route, evaluations=evaluations)

def run_guarded_response(guardrail_input, config=None, *, semantic_checker=None,
        evidence_checker=None, rewriter=None, fallback_registry=None,
        approved_templates=None, rewrite_timeout_seconds=10.0):
    """All external checkers and rewrites are isolated; production defaults abstain."""
    context, evaluations = None, 0
    registry = FallbackRegistry()
    mapper = DecisionMapper()
    started = monotonic()
    try:
        initial = ContractValidator().validate(guardrail_input)
        if initial['status'] != 'OK':
            return _blocked_delivery(guardrail_input, mapper.failure(guardrail_input, initial['status']), registry, evaluations)
        context = deepcopy(guardrail_input)
        registry = FallbackRegistry() if fallback_registry is None else fallback_registry
        if type(registry) is not FallbackRegistry:
            raise ValueError('Invalid fallback registry')
        runtime = RuntimeConfig() if config is None else config
        if type(runtime) is not RuntimeConfig:
            raise ValueError('Invalid runtime')
        deadline = started + runtime.request_timeout_seconds
        # Availability is derived from trusted objects, never a candidate or route flag alone.
        runtime = replace(runtime, approved_fallback_routes=registry.available_routes,
            controlled_rewrite_available=runtime.controlled_rewrite_available and rewriter is not None)
        mapper = DecisionMapper(runtime)
        semantic = None if semantic_checker is None else BoundedChecker(
            semantic_checker, 'semantic_safety', runtime.semantic_checker_timeout_seconds)
        evidence = None if evidence_checker is None else BoundedChecker(
            evidence_checker, 'evidence_checker', runtime.evidence_checker_timeout_seconds)
        while True:
            result = run_output_guardrail(context, runtime, semantic_checker=semantic,
                evidence_checker=evidence, approved_templates=approved_templates,
                rewriter=rewriter, fallback_registry=registry, deadline=deadline)
            evaluations += 1
            if not valid_final_result(result, context, runtime):
                result = mapper.failure(context)
            if monotonic() > deadline:
                result = mapper.failure(context, 'TIMEOUT', result['reason_codes'])
            if result['decision'] == 'ALLOW':
                return DeliveryResult(deepcopy(result), 'DISPLAY', result['final_response'],
                    context['system_route'], evaluations=evaluations)
            if result['action'] != 'REWRITE':
                return _blocked_delivery(context, result, registry, evaluations)
            try:
                rewrite_context = deepcopy(context)
                rewrite_context['verification_evidence'] = [e for e in rewrite_context['verification_evidence']
                    if e['review_status'] == 'approved']
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise ExecutionFailure('TIMEOUT')
                rewritten = bounded_invoke(rewriter, 'rewrite',
                    (rewrite_context, deepcopy(result)), min(rewrite_timeout_seconds, remaining))
                context = prepare_rewrite_input(context, result, rewritten, runtime)
            except ExecutionFailure as exc:
                failed = mapper.failure(context, exc.status, result['reason_codes'])
                return _blocked_delivery(context, failed, registry, evaluations)
            except Exception:
                failed = mapper.failure(context, reason_codes=result['reason_codes'])
                return _blocked_delivery(context, failed, registry, evaluations)
    except Exception:
        failed = mapper.failure(context)
        return _blocked_delivery(context, failed, registry, evaluations)
