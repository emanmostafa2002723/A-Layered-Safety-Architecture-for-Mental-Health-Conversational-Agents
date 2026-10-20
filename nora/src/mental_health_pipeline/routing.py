"""Single deterministic precedence function for all downstream routing."""

from __future__ import annotations

from .schemas import BoundaryDecision, CombinedSafetyDecision, CrisisDecision, RequestScope, SystemRoute


def resolve_route(
    crisis: CrisisDecision,
    boundary: BoundaryDecision,
    *,
    scope: RequestScope,
) -> CombinedSafetyDecision:
    """Resolve once; later components are not allowed to downgrade this route."""

    degradation: list[str] = []
    if crisis.status != "ok":
        degradation.append(f"crisis_{crisis.status}")
    if crisis.truncated:
        degradation.append("crisis_input_truncated")
    if boundary.status != "ok":
        degradation.append(f"boundary_{boundary.status}")
    if not boundary.complete_input_seen:
        degradation.append("boundary_incomplete_input")

    if boundary.raw.get("requires_rebuild_routing_adapter"):
        from .boundary_policy import boundary_handling
        handling = boundary_handling(crisis, boundary)
        route = SystemRoute(handling["route"])
        # Adult-reporter child safeguarding and emergencies must not be erased
        # by an unsupported scope. Other unsupported requests stay restricted.
        if not scope.is_supported and route not in {SystemRoute.CRISIS, SystemRoute.URGENT_MEDICAL}:
            route = SystemRoute.UNSUPPORTED_SCOPE
            degradation.extend(scope.unsupported_reasons)
    elif crisis.tier == "crisis":
        route = SystemRoute.CRISIS
    elif boundary.suggested_route == "urgent_medical_escalation" or boundary.explicit_emergency:
        route = SystemRoute.URGENT_MEDICAL
    elif not scope.is_supported:
        route = SystemRoute.UNSUPPORTED_SCOPE
        degradation.extend(scope.unsupported_reasons)
    elif boundary.suggested_route == "safe_completion":
        route = SystemRoute.BOUNDARY_SAFE_COMPLETION
    elif crisis.tier == "monitored":
        route = SystemRoute.MONITORED
    else:
        route = SystemRoute.NORMAL

    # Contract failures cannot result in unrestricted generation even if a bad
    # upstream mapping claimed "allow".
    if degradation and route == SystemRoute.NORMAL:
        route = SystemRoute.BOUNDARY_SAFE_COMPLETION

    return CombinedSafetyDecision(
        route=route,
        crisis=crisis,
        boundary=boundary,
        degraded=bool(degradation),
        degradation_reasons=tuple(dict.fromkeys(degradation)),
    )
