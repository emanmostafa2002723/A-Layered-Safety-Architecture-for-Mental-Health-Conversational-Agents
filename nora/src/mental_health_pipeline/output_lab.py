"""Isolated output-development lab. Never changes the active pipeline backend.

The optional compact model can block its four scoped clinical hazards. It
cannot clear other hazards: a complete passing Qwen/rules judgment is still
required to display any original response or replacement.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time


DIAGNOSIS_NOTICE=("I can’t diagnose you or determine whether you have a particular condition. "
    "I can listen, share general mental-health information, and help you prepare questions for a qualified clinician.")
MEDICATION_NOTICE=("I can’t recommend a medicine, dose, or medication change. Please ask your prescriber or pharmacist "
    "about your situation. I can help you prepare questions for them.")
TREATMENT_NOTICE=("I can’t prescribe a personalized treatment plan. I can share general information about therapy "
    "options and help you prepare questions for a qualified clinician.")
GENERAL_NOTICE=("I couldn’t safely provide that response. I can offer general emotional support and help you "
    "consider appropriate professional support.")


def replacement_for(codes):
    codes=set(codes)
    if codes & {'MEDICATION_RECOMMENDATION','PERSONALIZED_MEDICATION_INSTRUCTION'}:
        if codes & {'PERSONALIZED_DIAGNOSIS','PERSONAL_CLINICAL_INTERPRETATION'}:
            return DIAGNOSIS_NOTICE+'\n\n'+MEDICATION_NOTICE
        return MEDICATION_NOTICE
    if codes & {'PERSONALIZED_DIAGNOSIS','PERSONAL_CLINICAL_INTERPRETATION'}:return DIAGNOSIS_NOTICE
    if 'INDIVIDUALIZED_TREATMENT_PLAN' in codes:return TREATMENT_NOTICE
    return GENERAL_NOTICE


def load_clinical_output_model(path):
    """Explicit advisory-runtime dispatch; never interpret V2 as V1 inference."""
    root=Path(path)
    calibration=json.loads((root/'calibration.json').read_text(encoding='utf-8'))
    if 'window_tokens' in calibration:
        from .paragraph_output_model import ParagraphOutputModel
        return ParagraphOutputModel(root)
    return ClinicalOutputModel(root)


class ClinicalOutputModel:
    """GPU-only prototype inference; no display-approval method exists."""
    def __init__(self,path):
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        from .output_training_data import LABELS
        self.torch=torch
        if not torch.cuda.is_available():raise RuntimeError('Output lab requires CUDA')
        root=Path(path)
        manifest=json.loads((root/'training_manifest.json').read_text(encoding='utf-8'))
        actual=hashlib.sha256((root/'model.safetensors').read_bytes()).hexdigest()
        if manifest['model_sha256']!=actual or manifest['labels']!=list(LABELS):
            raise ValueError('Unpaired clinical checkpoint')
        calibration=json.loads((root/'calibration.json').read_text(encoding='utf-8'))
        if calibration!=manifest['calibration']:raise ValueError('Unpaired calibration')
        if 'window_tokens' in calibration:
            raise ValueError('Windowed checkpoints require the paragraph inference runtime')
        self.thresholds=calibration['thresholds'];self.labels=LABELS;self.model_sha256=actual
        self.tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(root,local_files_only=True).float().cuda().eval()

    def assess(self,text):
        torch=self.torch
        tokens=self.tokenizer(text,return_tensors='pt',truncation=False)
        if tokens['input_ids'].shape[1]>192:
            return {'status':'OUT_OF_SCOPE_LENGTH','codes':[],'no_display_authorization':True}
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            p=self.model(**{k:v.cuda() for k,v in tokens.items()}).logits.float().sigmoid().cpu()[0].tolist()
        torch.cuda.synchronize()
        return {'status':'ASSESSED','codes':[l for l,s,t in zip(self.labels,p,self.thresholds) if s>=t],
            'scores':dict(zip(self.labels,p)),'latency_ms':(time.perf_counter()-started)*1000,
            'model_sha256':self.model_sha256,'no_display_authorization':True}


class OutputGuardLab:
    def __init__(self,judge,clinical_model=None,*,use_compact_blocks=False):
        self.judge=judge;self.clinical_model=clinical_model
        # Broader diagnostic failures prevent default enforcement of this head.
        self.use_compact_blocks=use_compact_blocks

    def review(self,payload):
        started=time.perf_counter()
        try:
            return self._review(payload)
        except Exception as error:
            # A failed advisory model/transport must not leave an old GUI answer
            # looking like the current candidate's approved replacement.
            return {'schema_version':'output-lab-result/1.0','action':'ABSTAIN',
                'displayed_response':None,'pipeline_integration_enabled':False,
                'clinical_deployment_ready':False,'compact_enforcement':self.use_compact_blocks,
                'original_verdict':{'action':'ABSTAIN','reason_codes':[],
                    'error_code':'output_lab_execution_failed'},
                'replacement_verdict':None,'error_type':type(error).__name__,
                'total_latency_ms':(time.perf_counter()-started)*1000}

    def _review(self,payload):
        started=time.perf_counter();compact=None
        if self.clinical_model:
            compact=self.clinical_model.assess(payload['candidate'])
        if self.use_compact_blocks and compact and compact['codes']:
            original={'action':'ABSTAIN','reason_codes':compact['codes'],'error_code':None,
                'diagnostics':{'blocking_stage':'experimental_clinical_model'}}
        else:
            # Clear/unknown compact predictions cannot bypass any other check.
            original=self.judge.assess(payload)
        response={'schema_version':'output-lab-result/1.0','clinical_deployment_ready':False,
            'pipeline_integration_enabled':False,'compact_enforcement':self.use_compact_blocks,
            'compact':compact,'original_verdict':original,
            'original_sha256':hashlib.sha256(payload['candidate'].encode()).hexdigest(),
            'displayed_response':None,'action':'ABSTAIN','replacement_verdict':None}
        if original.get('action')=='DISPLAY' and not original.get('error_code') and not original.get('reason_codes'):
            response.update(action='DISPLAY',displayed_response=payload['candidate'])
        elif original.get('reason_codes') and not original.get('error_code'):
            new_payload=deepcopy(payload)
            new_payload['candidate']=replacement_for(original['reason_codes'])
            checked=self.judge.assess(new_payload)
            response['replacement_verdict']=checked
            if checked.get('action')=='DISPLAY' and not checked.get('error_code') and not checked.get('reason_codes'):
                response.update(action='DISPLAY_CHECKED_REPLACEMENT',displayed_response=new_payload['candidate'])
        response['total_latency_ms']=(time.perf_counter()-started)*1000
        return response
