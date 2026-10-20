"""Synthetic contract/adapter tests, never evidence of clinical model accuracy."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from mental_health_pipeline.boundary_policy import IntegratedPolicyBuilder
from mental_health_pipeline.config import PipelineConfig, RuntimeMode
from mental_health_pipeline.guardrails import bridge_static_crisis
from mental_health_pipeline.output_guard import PlaceholderOutputGuard, current_first_person_safety_disclosure
from mental_health_pipeline.output_integration import StandaloneOutputGate, frozen_input
from mental_health_pipeline.response_policy import render_response_policy
from mental_health_pipeline.routing import resolve_route
from mental_health_pipeline.schemas import BoundaryDecision, CrisisDecision, RequestScope
from mental_health_pipeline.worker_client import JsonLineWorker, WorkerConfig

ROOT = Path(__file__).resolve().parents[1]


def combined(kind='allow', reasons=(), signals=(), crisis=False):
    c = CrisisDecision.from_mapping(bridge_static_crisis({'schema_version': 'crisis-static-v2',
        'action': 'crisis_specific_route' if crisis else 'standard_response_eligible',
        'detected_categories': ['suicidal_ideation'] if crisis else []}))
    b = BoundaryDecision.from_mapping({'schema_version': 'boundary-rebuild-1.0',
        'requires_rebuild_routing_adapter': True, 'decision_kind': kind,
        'reasons': list(reasons), 'safety_signals': list(signals), 'status': 'ok',
        'suggested_route': 'allow' if kind == 'allow' else 'safe_completion', 'labels': []})
    return resolve_route(c, b, scope=RequestScope(age_eligibility_18_plus=True))


@pytest.mark.parametrize('kind,reasons,signals,route,guidance', [
    ('allow', (), (), 'normal_chat', 'normal_support_v1'),
    ('restrict', ('medication_decision',), (), 'clinical_boundary_safe_completion', 'final_boundary_restricted_v1'),
    ('uncertain', (), (), 'clinical_boundary_safe_completion', 'final_boundary_clarify_v1'),
    ('allow', (), ('child_safeguarding',), 'crisis_response', 'final_boundary_child_v1'),
    ('restrict', ('diagnosis',), ('medical_emergency',), 'urgent_medical_response', 'urgent_medical_response_v1'),
])
def test_boundary_route_and_instructions_share_authority(kind, reasons, signals, route, guidance):
    c = combined(kind, reasons, signals)
    p = IntegratedPolicyBuilder().build(c.to_policy_dict())
    assert c.route.value == p.system_route.value == route
    assert p.generator_guidance_id == guidance
    rendered = render_response_policy(p)
    if guidance == 'final_boundary_child_v1':
        assert 'child' in rendered and 'boundary_limit_notice' not in p.required_visible_elements
    if guidance == 'final_boundary_clarify_v1':
        assert 'Uncertainty is not itself a medical request' in rendered
        assert 'boundary_limit_notice' not in p.required_visible_elements


def test_medical_priority_retains_actual_crisis_evidence():
    c = combined(signals=('medical_emergency',), crisis=True)
    assert c.route.value == 'urgent_medical_response'
    assert c.crisis.categories == ('suicidal_ideation',)
    assert c.to_policy_dict()['boundary_handling']['crisis_signal_retained']
    assert IntegratedPolicyBuilder().build(c.to_policy_dict()).system_route.value == c.route.value


def test_pins_paths_and_production_gate():
    cfg = PipelineConfig.from_env(project_root=ROOT, environ={})
    assert cfg.crisis_weights.name == 'epoch_3.pt'
    assert cfg.crisis_backend == 'fresh_epoch3'
    assert cfg.crisis_calibration.name == 'calibration.json'
    assert cfg.mistral_adapter == ROOT / 'models/phoenix_v6_safe_lora'
    assert cfg.boundary_root == ROOT / 'components/boundary'
    assert cfg.output_guard_backend == 'standalone'
    assert 'production_candidate_not_accepted' in replace(cfg, runtime_mode=RuntimeMode.PRODUCTION).readiness_issues()


@pytest.mark.parametrize('text', [
    'I have been having thoughts about suicide, but I do not plan to act on them.',
    'I am thinking about suicide.', 'I keep having suicidal thoughts.',
    'I want to end my life.', 'I am safe now but I plan to end my life tonight.',
])
def test_current_disclosure_paraphrases_need_safety_question(text):
    p = IntegratedPolicyBuilder().build(combined(crisis=True).to_policy_dict())
    p = replace(p, required_visible_elements=())  # isolate the question requirement from contact rendering
    verdict = PlaceholderOutputGuard().inspect_with_context('Is there someone you can talk to right now?', p, user_text=text)
    assert 'missing_immediate_safety_question' in verdict.reason_codes


@pytest.mark.parametrize('text', [
    'What are suicidal thoughts?', 'My friend has thoughts about suicide.',
    'I do not have suicidal thoughts.', 'I had suicidal thoughts years ago.',
    'I am studying suicidal thoughts for my essay.', 'Translate "I want to end my life" into French.',
    'hi', 'I feel nervous about an exam.',
])
def test_mentions_not_asserted_as_current_personal_intent(text):
    assert not current_first_person_safety_disclosure(text)


class EchoWorker:
    def __init__(self, mode='DISPLAY'):
        self.mode, self.payload = mode, None
    def request(self, operation, payload):
        self.payload = payload
        f = payload['frozen_input']
        return {'action': self.mode, 'text': f['generated_response'] if self.mode == 'DISPLAY' else None,
                'route': f['system_route']}


def test_final_gate_accepts_only_identical_checked_text():
    p = IntegratedPolicyBuilder().build(combined().to_policy_dict())
    worker = EchoWorker()
    assert StandaloneOutputGate(worker).deliver('Hello.', p)['text'] == 'Hello.'
    worker.mode = 'ABSTAIN'
    assert StandaloneOutputGate(worker).deliver('Private draft.', p)['text'] is None
    class Changed(EchoWorker):
        def request(self, *args):
            return {'action': 'DISPLAY', 'text': 'Changed after approval', 'route': 'normal_chat'}
    assert StandaloneOutputGate(Changed()).deliver('Hello.', p)['action'] == 'ABSTAIN'


def test_frozen_schema_and_unapproved_evidence():
    from mental_health_pipeline.rag import Citation, RetrievedDocument
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    p = IntegratedPolicyBuilder().build(combined().to_policy_dict())
    doc = RetrievedDocument(text='A synthetic evidence passage.', score=1,
        citation=Citation(source_id='source-1', title='Synthetic', source='Test'))
    f = frozen_input('Hello.', p, SimpleNamespace(documents=(doc,)))
    assert f['verification_evidence'][0]['review_status'] == 'pending'
    assert f['response_policy']['factual_verification']['required']
    assert f['system_route'] == p.system_route.value
    schemas = [json.loads(path.read_text(encoding='utf-8'))
               for path in (ROOT / 'components/output_guardrails/schemas').glob('*.json')]
    registry = Registry().with_resources((s['$id'], Resource.from_contents(s)) for s in schemas)
    schema = next(s for s in schemas if s['$id'].endswith('/input_schema.json'))
    Draft202012Validator(schema, registry=registry).validate(f)


def test_real_standalone_service_defaults_abstain_without_network():
    # Source contract probe on local Python; target Python 3.12 acceptance is separate.
    worker = JsonLineWorker(WorkerConfig(component='output', python_executable=Path(sys.executable),
        module='mental_health_pipeline.workers.output', arguments=(
            '--component-root', str(ROOT / 'components/output_guardrails'),
            '--provider-config', str(ROOT / 'config/output_providers.json')),
        source_root=ROOT / 'src', startup_timeout_seconds=20, request_timeout_seconds=20))
    try:
        p = IntegratedPolicyBuilder().build(combined().to_policy_dict())
        result = worker.request('deliver', {'frozen_input': frozen_input('Synthetic check.', p)})
        assert result['action'] == 'ABSTAIN' and result['text'] is None
    finally:
        worker.close()


def test_weights_only_load_is_strict_and_hash_bound(tmp_path):
    import torch
    from mental_health_pipeline.new_components import load_weights_only
    model = torch.nn.Linear(2, 1)
    state = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
    path = tmp_path / 'final_weights.pt'
    torch.save(state, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    load_weights_only(model, path, digest)
    assert torch.equal(model.weight, state['weight'])
    with pytest.raises(ValueError):
        load_weights_only(model, path, '0'*64)
    torch.save({'wrong': torch.ones(2)}, path)
    with pytest.raises(ValueError):
        load_weights_only(model, path, hashlib.sha256(path.read_bytes()).hexdigest())
