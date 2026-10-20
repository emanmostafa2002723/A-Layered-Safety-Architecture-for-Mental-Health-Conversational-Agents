"""Additional medical veto plus full Qwen/rules review; development only.

The medical head is not a complete guard and cannot authorize display. Missed
medical risks still reach the full judge. Every replacement is checked again
with the original route, requirements and evidence preserved.
"""
import hashlib
import json
from pathlib import Path
import time

from .local_output import LocalJudge, LocalQwenOutputGate, JudgeSettings, VERSION
from .output_lab import replacement_for
from .output_training_data import LABELS


def validate_medical_assets(settings_path, project_root):
    root=Path(project_root).resolve()
    settings=json.loads(Path(settings_path).read_text(encoding='utf-8'))
    JudgeSettings.load(settings_path)
    if (settings.get('medical_mode')!='additional_veto_full_judge_for_clear'
            or settings.get('medical_device')!='cuda:0'
            or settings.get('standalone_medical_model_accepted') is not False):
        raise ValueError('Unsupported medical runtime or classifier-only approval')
    model=(root/settings['medical_model']).resolve()
    model.relative_to(root)
    for name,key in (('model.safetensors','medical_model_sha256'),
                     ('calibration.json','medical_calibration_sha256')):
        with (model/name).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=settings[key]:
                raise ValueError('Pinned medical assets changed')
    return model


class HybridJudge:
    def __init__(self, judge, medical_model):
        self.judge, self.medical_model = judge, medical_model

    @classmethod
    def load(cls, component_root, settings_path, project_root):
        from .paragraph_output_model import ParagraphOutputModel
        model = validate_medical_assets(settings_path,project_root)
        medical = ParagraphOutputModel(model)
        return cls(LocalJudge(component_root, JudgeSettings.load(settings_path)), medical)

    def assess(self, payload):
        started = time.perf_counter()
        try:
            medical = self.medical_model.assess(payload['candidate'])
            codes = medical.get('codes', [])
            if medical.get('status') != 'ASSESSED' or not set(codes) <= set(LABELS):
                raise ValueError('Medical coverage or label contract failed')
            if codes:
                return {'action':'ABSTAIN', 'reason_codes':codes, 'error_code':None,
                    'route':payload.get('system_route'), 'diagnostics':{
                        'guard_version':VERSION, 'medical':medical,
                        'model_called':False, 'blocking_stage':'additional_medical_veto',
                        'clinical_deployment_ready':False,
                        'latency_ms':(time.perf_counter()-started)*1000}}
            result = self.judge.assess(payload)
            return {**result, 'diagnostics':{**result.get('diagnostics',{}),
                'medical':medical, 'hybrid_total_latency_ms':(time.perf_counter()-started)*1000}}
        except Exception:
            return {'action':'ABSTAIN', 'reason_codes':[],
                'error_code':'hybrid_medical_check_failed',
                'route':payload.get('system_route') if isinstance(payload,dict) else None,
                'diagnostics':{'guard_version':VERSION, 'clinical_deployment_ready':False}}


class HybridOutputGate(LocalQwenOutputGate):
    """Only a fully checked medical-limit replacement may replace a draft."""
    def deliver(self, text, packet, retrieval=None):
        original = super().deliver(text, packet, retrieval)
        if original.get('action') == 'DISPLAY':
            return original
        codes = original.get('reason_codes', [])
        # Do not mask an unrelated safety issue, transport error or uncertainty
        # by replacing the response with a medical notice.
        if (not codes or not set(codes) <= set(LABELS) or original.get('error_code')):
            return original
        replacement = replacement_for(codes)
        checked = super().deliver(replacement, packet, retrieval)
        if checked.get('action') != 'DISPLAY':
            return {**checked, 'action':'ABSTAIN', 'text':None,
                'original_reason_codes':codes, 'replacement_checked':False}
        return {**checked, 'action':'DISPLAY_CHECKED_REPLACEMENT',
            'replacement_checked':True, 'original_reason_codes':codes,
            'original_medical_check':original.get('diagnostics',{}).get('medical'),
            'original_blocking_stage':original.get('diagnostics',{}).get('blocking_stage'),
            'original_sha256':hashlib.sha256(text.encode()).hexdigest()}
