"""Canonical routes; no aliases or route selection."""
SUPPORTED_ROUTES = frozenset({'normal_chat', 'monitored_support', 'clinical_boundary_safe_completion', 'urgent_medical_response', 'crisis_response'})
URGENT_ROUTES = frozenset({'urgent_medical_response', 'crisis_response'})
# Upstream must explicitly provide these minimum route constraints.
ROUTE_REQUIREMENTS = {
    'normal_chat': frozenset(), 'monitored_support': frozenset(),
    'clinical_boundary_safe_completion': frozenset({'CLINICAL_BOUNDARY_STATEMENT'}),
    'urgent_medical_response': frozenset({'URGENT_ESCALATION', 'SAFETY_ORIENTED_GUIDANCE'}),
    'crisis_response': frozenset({'CRISIS_SUPPORT', 'URGENT_ESCALATION', 'SAFETY_ORIENTED_GUIDANCE'}),
}

def required_behaviors(route, policy):
    """One calculation used by required-safety checking and independent fusion."""
    return set(policy['required_behaviors']) | ROUTE_REQUIREMENTS[route]


def route_requirements_missing(route, policy):
    """Whether upstream omitted a mandatory route requirement from its policy."""
    return not ROUTE_REQUIREMENTS[route] <= set(policy['required_behaviors'])


def clinical_boundary_policy_weakened(route, policy):
    """The clinical route must explicitly prohibit personalized conclusions."""
    return route == 'clinical_boundary_safe_completion' and not policy['personalized_conclusions_prohibited']
