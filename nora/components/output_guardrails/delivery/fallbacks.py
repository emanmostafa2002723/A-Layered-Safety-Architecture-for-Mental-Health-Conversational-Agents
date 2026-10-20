"""Trusted static template registry. Production registry is empty by default."""
from dataclasses import dataclass
from types import MappingProxyType
from baseline.deterministic_checker import DeterministicChecker
from baseline.contracts import valid_policy_context, initialize_contracts
from config.routes import SUPPORTED_ROUTES
from validators.route_policy_validator import RoutePolicyValidator
from validators.required_safety_validator import RequiredSafetyValidator

@dataclass(frozen=True)
class FallbackTemplate:
    route: str
    template_id: str
    version: str
    text: str
    approval_id: str
    approved: bool = False

    def __post_init__(self):
        if type(self.route) is not str or self.route not in SUPPORTED_ROUTES:
            raise ValueError('Invalid fallback route')
        if any(type(v) is not str or not v.strip() or len(v) > 256
               for v in (self.template_id, self.version, self.approval_id)):
            raise ValueError('Invalid fallback identity')
        if type(self.text) is not str or not self.text.strip() or len(self.text) > 20000:
            raise ValueError('Invalid fallback text')
        if type(self.approved) is not bool or not self.approved:
            raise ValueError('Explicit fallback approval required')

class FallbackRegistry:
    def __init__(self, templates=()):
        entries = {}
        for template in templates:
            if type(template) is not FallbackTemplate or template.route in entries:
                raise ValueError('Invalid or duplicate fallback registration')
            entries[template.route] = template
        self._templates = MappingProxyType(entries)

    @property
    def available_routes(self):
        return frozenset(self._templates)

    def resolve(self, route, policy):
        """Resolve only static registered text compatible with this route/policy."""
        try:
            if type(route) is not str:
                return None
            template = self._templates.get(route)
            if template is None or not valid_policy_context(template.text, route, policy):
                return None
            initialize_contracts()
            safety = DeterministicChecker().check(template.text, {'system_route': route})
            routing = RoutePolicyValidator({template.template_id: template.text}).validate(
                template.text, route, policy, safety['violations'])
            required = RequiredSafetyValidator().validate(template.text, route, policy)
            if any(r['status'] not in {'OK', 'NOT_APPLICABLE'} or r['violations']
                   for r in (safety, routing, required)):
                return None
            return template
        except Exception:
            return None
