"""Application display boundary. Candidate text never goes directly to a sink.

Routing/generation/RAG remain upstream. This adapter owns only the transition
from a frozen guardrail input to one complete, validated display message.
"""
from copy import deepcopy
from dataclasses import dataclass
from baseline.contracts import input_contract_error, validate_schema, valid_final_result
from config.routes import SUPPORTED_ROUTES
from config.fusion_policy import REWRITE_ROUTES
from config.runtime import RuntimeConfig
from delivery.pipeline import DeliveryResult
from delivery.fallbacks import FallbackRegistry
from integration.service import GuardrailService


@dataclass(frozen=True)
class UserResponse:
    action: str
    text: object
    route: object


def _validated_delivery_response(result, context, *, runtime=None, registry=None, rewriter=None):
    """One delivery validation boundary shared with observation-only evaluation.

    This verifies authorization/identity, not detector accuracy or a new model call.
    Components and configuration remain trusted deployment dependencies.
    """
    runtime = RuntimeConfig() if runtime is None else runtime
    registry = FallbackRegistry() if registry is None else registry
    if (type(runtime) is not RuntimeConfig or type(registry) is not FallbackRegistry
            or input_contract_error(context) is not None):
        raise ValueError('Invalid delivery context')
    route = context['system_route']
    if type(result) is not DeliveryResult or result.route != route:
        raise ValueError('Invalid delivery identity')
    guardrail = result.guardrail_result
    if not validate_schema('decision_schema.json', guardrail):
        raise ValueError('Invalid delivery contract')
    if result.delivery_action == 'DISPLAY':
        if (guardrail['decision'], guardrail['action']) != ('ALLOW', 'DISPLAY'):
            raise ValueError('Display was not authorized')
        if result.delivered_text != guardrail['final_response']:
            raise ValueError('Display text changed after validation')
        remaining_attempts = max(0, runtime.max_rewrite_attempts
            - context['execution_context']['rewrite_attempt'])
        if (type(result.evaluations) is not int or not 1 <= result.evaluations <= remaining_attempts + 1):
            raise ValueError('Invalid evaluation history')
        if result.evaluations == 1 and result.delivered_text != context['generated_response']:
            raise ValueError('Original candidate identity changed')
        if result.evaluations > 1 and (not runtime.controlled_rewrite_available
                or rewriter is None or route not in REWRITE_ROUTES):
            raise ValueError('Rewrite history was not authorized')
        checked_context = deepcopy(context)
        checked_context['generated_response'] = result.delivered_text
        checked_context['execution_context']['rewrite_attempt'] += result.evaluations - 1
        if not valid_final_result(guardrail, checked_context, runtime):
            raise ValueError('Contextual display authorization failed')
        text = result.delivered_text
    elif result.delivery_action == 'SAFE_FALLBACK':
        if (guardrail['decision'], guardrail['action']) != ('BLOCK', 'SAFE_FALLBACK'):
            raise ValueError('Fallback was not authorized')
        template = registry.resolve(route, context['response_policy'])
        if (template is None or (result.template_id, result.template_version, result.delivered_text)
                != (template.template_id, template.version, template.text)):
            raise ValueError('Fallback identity changed')
        text = template.text
    elif (result.delivery_action == 'ABSTAIN' and guardrail['decision'] == 'BLOCK'
            and guardrail['action'] == 'ABSTAIN' and result.delivered_text is None):
        return UserResponse('ABSTAIN', None, route)
    else:
        raise ValueError('Intermediate actions cannot reach display')
    if type(text) is not str or not text.strip() or len(text) > 20000:
        raise ValueError('Invalid display text')
    return UserResponse(result.delivery_action, text, route)


class GuardedApplication:
    def __init__(self, service):
        if type(service) is not GuardrailService:
            raise ValueError('A configured guardrail service is required')
        self.service = service

    def respond(self, frozen_input):
        """Return display data only; failure details and candidates stay internal."""
        route = None
        try:
            if input_contract_error(frozen_input) is not None:
                return UserResponse('ABSTAIN', None, None)
            context = deepcopy(frozen_input)
            route = context['system_route']
            result = self.service.process(deepcopy(context))
            return _validated_delivery_response(result, context, runtime=self.service.runtime,
                registry=self.service.registry, rewriter=self.service.rewriter)
        except Exception:
            return UserResponse('ABSTAIN', None, route if route in SUPPORTED_ROUTES else None)

    def deliver(self, frozen_input, display):
        """The sink is called once, after validation; no candidate streaming exists.

        A sink failure propagates. It is never retried with the rejected candidate.
        The caller owns rendering/access control and must not add another display path.
        """
        if not callable(display):
            raise ValueError('A trusted display sink is required')
        response = self.respond(frozen_input)
        if response.text is not None:
            display(response.text)
        return response
