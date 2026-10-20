"""Map application policy/evidence to the frozen standalone output contract.

No provider is auto-approved, no RAG source is auto-reviewed, no route is aliased.
The final display gate also checks application fallback/contact text.
"""
import hashlib
import json
from pathlib import Path


REQUIRED = {
    'normal_chat': [], 'monitored_support': [],
    'clinical_boundary_safe_completion': ['CLINICAL_BOUNDARY_STATEMENT'],
    'urgent_medical_response': ['URGENT_ESCALATION', 'SAFETY_ORIENTED_GUIDANCE'],
    'crisis_response': ['CRISIS_SUPPORT', 'URGENT_ESCALATION', 'SAFETY_ORIENTED_GUIDANCE'],
}


def frozen_input(response, packet, retrieval=None, governed_sources=None):
    route = packet.system_route.value
    if route not in REQUIRED:
        raise ValueError('Output contract does not support this route; must abstain')
    evidence = []
    for doc in (retrieval.documents if retrieval else ()):
        content_hash = hashlib.sha256(doc.text.encode()).hexdigest()
        reviewed = (governed_sources or {}).get(content_hash, {})
        evidence.append({'evidence_id': content_hash, 'source_id': doc.citation.source_id,
            'source_version': content_hash, 'text': doc.text[:20000],
            'review_status': 'approved' if reviewed.get('review_status') == 'approved' else 'pending',
            'metadata': {'title': doc.citation.title[:512],
                         'review_record': str(reviewed.get('review_record', 'not_supplied'))[:512]}})
    return {'schema_version': 'output-guardrail-input-1.0', 'generated_response': response,
        'system_route': route, 'response_policy': {
            'policy_version': 'response-policy-1.0', 'template_id': None,
            'may_continue_with_general_information': packet.generation_allowed,
            'personalized_conclusions_prohibited': True,
            'prohibited_behaviors': ['PERSONALIZED_DIAGNOSIS', 'MEDICATION_RECOMMENDATION',
                'PERSONALIZED_MEDICATION_INSTRUCTION', 'INDIVIDUALIZED_TREATMENT_PLAN',
                'HARMFUL_INSTRUCTIONS', 'PRIVACY_LEAK', 'SYSTEM_PROMPT_LEAKAGE'],
            'required_behaviors': list(REQUIRED[route]),
            'factual_verification': {'required': bool(evidence) or route == 'normal_chat'}},
        'verification_evidence': evidence, 'execution_context': {'rewrite_attempt': 0}}


class StandaloneOutputGate:
    production_ready = False

    def __init__(self, worker, governed_sources_path=None):
        self.worker = worker
        self.sources = {}
        if governed_sources_path:
            self.sources = json.loads(Path(governed_sources_path).read_text(encoding='utf-8'))['sources']

    def deliver(self, text, packet, retrieval=None):
        try:
            payload = frozen_input(text, packet, retrieval, self.sources)
            result = self.worker.request('deliver', {'frozen_input': payload})
            # No rewriter/fallback registry is configured by this adapter.
            # Accept only the identical checked candidate returned by DISPLAY.
            if (result.get('action') == 'DISPLAY' and result.get('route') == payload['system_route']
                    and result.get('text') == text):
                return {'action': 'DISPLAY', 'text': text, 'route': result['route']}
            return {'action': 'ABSTAIN', 'text': None, 'reason': 'standalone_guard_withheld'}
        except Exception:
            return {'action': 'ABSTAIN', 'text': None, 'reason': 'standalone_guard_unavailable_or_contract_error'}

    def readiness(self):
        return {'ready': False, 'reason': 'provider_approval_and_real_model_acceptance_pending',
                'worker': self.worker.readiness(), 'production_ready': False}
