"""One authoritative route/policy bridge for the selected boundary release."""
from dataclasses import replace
from functools import lru_cache
import importlib.util
from pathlib import Path

from .response_policy import ResponsePolicyBuilder
from .response_policy.schemas import SystemRoute, Severity, TemporalStatus
from .response_policy.compose import BOUNDARY_PROHIBITIONS


@lru_cache(maxsize=1)
def _resolver():
    path = Path(__file__).resolve().parents[2] / 'components/boundary/boundary/integration.py'
    spec = importlib.util.spec_from_file_location('_selected_boundary_integration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve


def boundary_handling(crisis, boundary):
    # Pass the ACTUAL crisis decision, never a dummy normal value.
    return _resolver()(dict(crisis.raw), dict(boundary.raw))


def _retain_other_guidance(packet, primary):
    return tuple(dict.fromkeys(item for item in
        (packet.generator_guidance_id, *packet.secondary_guidance_ids)
        if item not in {'normal_support_v1', primary}))[:2]


class IntegratedPolicyBuilder(ResponsePolicyBuilder):
    def build(self, combined_decision, *, recent_state=None):
        packet = super().build(combined_decision, recent_state=recent_state)
        handling = combined_decision.get('boundary_handling')
        if not handling:
            return packet
        raw = combined_decision['clinical_boundary']
        guide = handling['guidance']
        # Use the SAME resolution as routing. In particular medical emergencies
        # take immediate priority, without dropping other prohibitions/evidence.
        canonical = combined_decision['system_route']
        if canonical == 'unsupported_scope':
            return packet
        changes = {'system_route': SystemRoute(canonical), 'policy_flags': tuple(dict.fromkeys(
            (*packet.policy_flags, 'selected_boundary_contract_applied', 'crisis_independent_acceptance_pending')))}
        mapping = {'diagnosis': 'diagnosis_or_clinical_interpretation',
                   'medication_decision': 'medication_recommendation_or_selection',
                   'clinical_treatment': 'personalized_treatment_plan',
                   'harmful_medical_instruction': 'harmful_or_unethical_medical_instruction'}
        selected_labels = tuple(sorted({mapping[x] for x in raw.get('reasons', ()) if x in mapping}))
        if selected_labels:
            # Crisis priority changes response framing, not the simultaneous
            # medical prohibitions. Do not prepend a routine refusal to an
            # emergency, but never lose these constraints in route resolution.
            changes.update(boundary_labels=selected_labels,
                prohibited_behaviors=tuple(dict.fromkeys((*packet.prohibited_behaviors,
                    *(item for label in selected_labels for item in BOUNDARY_PROHIBITIONS[label])))))
        if guide == 'urgent_medical':
            changes.update(generator_guidance_id='urgent_medical_response_v1',
                secondary_guidance_ids=_retain_other_guidance(packet,'urgent_medical_response_v1'),
                required_visible_elements=('worldwide_emergency_fallback', 'local_resources_if_eligible'),
                response_goals=('encourage_immediate_local_help', 'ask_about_urgent_medical_need'),
                severity=Severity.URGENT, temporal_status=TemporalStatus.CURRENT,
                fallback_template_id='urgent_medical_fallback_v1')
        elif guide == 'child_safeguarding':
            changes.update(generator_guidance_id='final_boundary_child_v1',
                secondary_guidance_ids=_retain_other_guidance(packet,'final_boundary_child_v1'),
                required_visible_elements=(), response_goals=('prioritize_child_safety', 'encourage_trusted_safe_adult'),
                fallback_template_id='trauma_support_fallback_v1')
        elif guide == 'clarify_or_general_support':
            changes.update(generator_guidance_id='final_boundary_clarify_v1', secondary_guidance_ids=(),
                boundary_labels=(), required_visible_elements=(), needs_clarification=True,
                response_goals=('respond_to_user_request',), fallback_template_id='supportive_unavailable_fallback_v1')
        elif guide == 'restricted_request':
            changes.update(generator_guidance_id='final_boundary_restricted_v1',
                secondary_guidance_ids=_retain_other_guidance(packet,'final_boundary_restricted_v1'),
                boundary_labels=selected_labels,
                prohibited_behaviors=tuple(dict.fromkeys((*packet.prohibited_behaviors,
                    *(item for label in selected_labels for item in BOUNDARY_PROHIBITIONS[label])))),
                required_visible_elements=('boundary_limit_notice',))
        # An existing recent genuine concern must still survive ordinary support.
        elif guide in {'ordinary_support_or_information', 'support_with_safety_check'}:
            changes['system_route'] = packet.system_route
        return replace(packet, **changes)
