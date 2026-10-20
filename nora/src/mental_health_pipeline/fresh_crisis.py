"""Portable epoch-3 inference with the supplied presence-gated calibration.

The original experiment's base-cache paths are provenance, never runtime inputs.
Instantiate its exact architecture from local config, then replace ALL tensors.
"""
import json
import math
from pathlib import Path
import threading

from .guardrails import bridge_static_crisis, sha256_file, ComponentStatus
from .new_components import load_weights_only

LABELS = ('suicidal_ideation', 'self_directed_harm', 'sexual_violence',
          'domestic_violence', 'child_abuse_or_endangerment')
WEIGHTS_SHA256 = '82bb3befdebd3311833a14b299a921cbd120764f6b9dd75766fd61112c61bf68'
CALIBRATION_SHA256 = '969101aee690025edfec12344b5f5980c3fb66a1b863ec50d87e4c4de9d7493d'


def read_calibration(path, weights_hash, calibration_hash):
    if not calibration_hash or sha256_file(path) != calibration_hash:
        raise ValueError('Fresh calibration hash mismatch or unpinned policy')
    policy = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if (policy.get('routing') != 'presence_gate_no_category_override'
            or policy.get('weights_sha256') != weights_hash
            or policy.get('selected_epoch') != 3
            or policy.get('clinical_deployment_authorized') is not False):
        raise ValueError('Fresh weights/calibration contract mismatch')
    temperatures = [policy.get('presence_temperature'), policy.get('type_temperature')]
    thresholds = policy.get('type_thresholds')
    if not isinstance(thresholds, list) or len(thresholds) != len(LABELS):
        raise ValueError('Expected five type thresholds in the reference label order')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in temperatures):
        raise ValueError('Invalid calibration temperature')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for v in [policy.get('presence_threshold'), *thresholds]):
        raise ValueError('Invalid calibration threshold')
    return policy


def decode_logits(presence_logits, type_logits, policy):
    """Numerically equivalent to reference evaluation.decode; no category veto."""
    import numpy as np
    p = np.asarray(presence_logits, dtype=float) / policy['presence_temperature']
    b = np.asarray(type_logits, dtype=float) / policy['type_temperature']
    if p.ndim != 2 or b.ndim != 2 or p.shape != (len(b), 2) or b.shape[1] != 5:
        raise ValueError('Unexpected logits shape')
    if not np.isfinite(p).all() or not np.isfinite(b).all():
        raise ValueError('Non-finite classifier logits')
    exp = np.exp(p - p.max(-1, keepdims=True))
    q = exp[:, 1] / exp.sum(-1)
    scores = np.exp(-np.logaddexp(0, -b))
    presence = q >= policy['presence_threshold']
    raw_types = scores >= np.asarray(policy['type_thresholds'])
    categories = raw_types & presence[:, None]
    return presence, categories, q, scores


def make_model(config_directory):
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel
    class FreshModel(nn.Module):
        def __init__(self):
            super().__init__()
            cfg = AutoConfig.from_pretrained(config_directory, local_files_only=True, trust_remote_code=False)
            if cfg.model_type != 'deberta-v2' or cfg.hidden_size != 768 or cfg.num_hidden_layers != 12:
                raise ValueError('Unexpected fresh encoder architecture')
            self.encoder = AutoModel.from_config(cfg, trust_remote_code=False).float()
            self.dropout = nn.Dropout(.1)
            self.presence = nn.Linear(cfg.hidden_size, 2)
            self.broad = nn.Linear(cfg.hidden_size, 5)
        def forward(self, **tokens):
            hidden = self.dropout(self.encoder(**tokens).last_hidden_state[:, 0])
            return {'presence': self.presence(hidden), 'broad': self.broad(hidden)}
    return FreshModel()


class FreshCrisisAdapter:
    version = 'crisis-fresh-epoch3-presence-gated/2026-10-19'

    def __init__(self, *, component_root, weights, calibration, weights_sha256=WEIGHTS_SHA256,
                 calibration_sha256=CALIBRATION_SHA256, device='cpu'):
        self.root = Path(component_root).resolve()
        self.weights, self.calibration_path = Path(weights).resolve(), Path(calibration).resolve()
        for p in (self.weights, self.calibration_path):
            p.relative_to(self.root)
        self.weights_sha256, self.calibration_sha256 = weights_sha256, calibration_sha256
        self.device = device
        self.model = self.tokenizer = self.policy = None
        self._load_lock, self._inference_lock = threading.Lock(), threading.Lock()

    def readiness(self):
        files = [self.weights, self.calibration_path, self.root/'tokenizer/config.json',
                 self.root/'tokenizer/tokenizer.json', self.root/'tokenizer/tokenizer_config.json']
        return ComponentStatus('crisis_guardrail', all(p.is_file() for p in files), self.version)

    def _load(self):
        if self.model is not None:
            return self
        with self._load_lock:
            if self.model is None:
                import torch
                from transformers import AutoTokenizer
                policy = read_calibration(self.calibration_path, self.weights_sha256, self.calibration_sha256)
                if self.device.startswith('cuda') and not torch.cuda.is_available():
                    raise RuntimeError('Requested CUDA is unavailable')
                tokenizer = AutoTokenizer.from_pretrained(self.root/'tokenizer', local_files_only=True,
                    trust_remote_code=False)
                model = make_model(self.root/'tokenizer')
                load_weights_only(model, self.weights, self.weights_sha256)
                model.float().to(self.device).eval()
                self.policy, self.tokenizer, self.model = policy, tokenizer, model
        return self

    def predict_batch(self, texts, *, batch_size=4):
        if isinstance(texts, (str, bytes)):
            raise TypeError('Expected message list')
        texts = list(texts)
        if (not texts or any(not isinstance(t, str) or not t.strip() for t in texts)
                or type(batch_size) is not int or batch_size < 1):
            raise ValueError('Invalid message batch')
        self._load()
        import torch
        results = []
        with self._inference_lock, torch.inference_mode():
            for start in range(0, len(texts), batch_size):
                chunk = texts[start:start+batch_size]
                counts = [len(ids) for ids in self.tokenizer(chunk, truncation=False, padding=False)['input_ids']]
                tokens = self.tokenizer(chunk, padding=True, truncation=True, max_length=512, return_tensors='pt')
                output = self.model(**{k:v.to(self.device) for k,v in tokens.items()
                                      if k in ('input_ids','attention_mask','token_type_ids')})
                gate, flags, q, scores = decode_logits(output['presence'].float().cpu().numpy(),
                    output['broad'].float().cpu().numpy(), self.policy)
                for positive, labels, probability, raw, count in zip(gate, flags, q, scores, counts):
                    results.append({'schema_version':'crisis-static-v2',
                        'model_id':'fresh_pretrained_full_encoder_epoch3_seed42',
                        'presence_label':'crisis_evidence' if positive else 'no_supported_crisis',
                        'presence_probabilities':{'no_supported_crisis':1-float(probability),'crisis_evidence':float(probability)},
                        'presence_threshold':self.policy['presence_threshold'],
                        'detected_categories':[k for k, flag in zip(LABELS,labels) if flag],
                        'category_probabilities':dict(zip(LABELS,map(float,raw))),
                        'category_thresholds':dict(zip(LABELS,self.policy['type_thresholds'])),
                        'action':'crisis_specific_route' if positive else 'standard_response_eligible',
                        'no_supported_crisis':not bool(positive), 'category_override':False,
                        'route_reasons':['presence_threshold'] if positive else [],
                        'decoding':'presence_gate_no_category_override',
                        'calibration_status':'paired_epoch3_development_fitted_not_clinically_validated',
                        'weights_sha256':self.weights_sha256,'calibration_sha256':self.calibration_sha256,
                        'temporal_status':'not_assessed','clinical_safety_status':'not_assessed',
                        'production_authorized':False,'truncated':count>512,'input_tokens':count,
                        'max_length':512,'device':self.device,'precision':'fp32'})
        return results

    def assess(self, text):
        return bridge_static_crisis(self.predict_batch([text],batch_size=1)[0])
