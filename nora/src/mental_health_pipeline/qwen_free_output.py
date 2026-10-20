"""Standalone Qwen-free development candidate; never clinical authorization.

The response-only head does not infer user intent or prove factual entailment.
Rules, explicit policy obligations and conservative extract support are separate
checks. Uncertain/unsupported claims are withheld, not guessed safe.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

from .qwen_free_data import LABELS, FEATURES, CODE_MAP
from .paragraph_output_model import probabilities
from .local_output import ROUTES, ELEMENTS, digest

VERSION='qwen-free-output/0.1-development'
MEDICAL=re.compile(r'\b(?:CBT|cognitive behavio\w*|anxiety|depress\w*|ADHD|psycho\w*|therap\w*|medicin\w*|medication\w*|drug\w*|symptom\w*|antidepress\w*|serotonin|panic|diagnos\w*|treatment\w*|disorder\w*|clinical\w*)\b',re.I)
REFUSAL=re.compile(r"\b(?:I|we) (?:can(?:not|'t)|am unable to|do not|don't) (?:diagnose|determine|choose|select|recommend|prescribe|interpret|assign|establish|infer|promise|guarantee|give|provide)\b",re.I)


def normalized(text):
    return ' '.join(re.findall(r"[a-z0-9]+(?:'[a-z]+)?",text.casefold().replace('’',"'")))


def validate_input(value):
    if (type(value) is not dict or set(value)!={'schema_version','candidate','system_route','policy','requirements','evidence'}
            or value['schema_version']!='local-output-input/1.0' or value['system_route'] not in ROUTES
            or type(value['candidate']) is not str or not value['candidate'].strip() or len(value['candidate'])>20000
            or type(value['policy']) is not dict or type(value['requirements']) is not list
            or len(value['requirements'])>12 or any(type(x) is not str for x in value['requirements'])
            or len(set(value['requirements']))!=len(value['requirements'])
            or not set(value['requirements'])<=set(ELEMENTS) or type(value['evidence']) is not list or len(value['evidence'])>8):
        raise ValueError('Invalid output guard payload')
    if value['policy'].get('route',value['system_route'])!=value['system_route']:
        raise ValueError('Policy route mismatch')
    if 'required_elements' in value['policy'] and value['policy']['required_elements']!=value['requirements']:
        raise ValueError('Policy requirements mismatch')
    ids=set()
    for row in value['evidence']:
        if (type(row) is not dict or set(row)!={'id','text','provenance'} or type(row['id']) is not str
                or not row['id'] or row['id'] in ids or type(row['text']) is not str or not row['text'].strip()
                or len(row['text'])>20000 or row['provenance'] not in {'retrieved_unreviewed','synthetic_test_only'}):
            raise ValueError('Invalid evidence')
        ids.add(row['id'])


def requirement_missing(text,requirements):
    """Conservative phrase checks, not semantic proof of crisis care quality."""
    from .output_guard import _required_visible_element_present, _normalized_for_match
    n=normalized(text)
    urgent=bool(re.search(r'\b(?:call|contact|seek|reach|go to)\b.{0,55}\b(?:emergency|urgent|immediate)\b|\b(?:emergency|urgent)\b.{0,45}\b(?:now|immediately)\b',n))
    if re.search(r'\b(?:do not|don t|never|avoid|delay|wait before)\b.{0,45}\b(?:call|contact|seek|help|emergency)\b',n):urgent=False
    affirmative=' '.join(clause for clause in re.split(r'[.!?;\n]',text)
        if not re.search(r"\b(?:not|never|cannot|can't|do not|don't|avoid|delay)\b",clause,re.I)
        or re.search(r'\bnot alone\b',clause,re.I))
    positive=normalized(affirmative)
    support=bool(re.search(r'\b(?:thank you|here (?:with|for|to listen)|can stay|can keep talking|sounds (?:hard|painful|difficult)|you (?:matter|deserve support)|not alone)\b',positive))
    safety=bool(re.search(r'\b(?:are you safe|(?:might|may) you act|safe right now|someone you trust|trusted person|distance between|away from anything)\b',positive))
    boundary=bool(REFUSAL.search(text))
    flags={'URGENT_ESCALATION':urgent,'CRISIS_SUPPORT':support,'SAFETY_ORIENTED_GUIDANCE':safety,
        'PROFESSIONAL_HELP_ENCOURAGEMENT':bool(re.search(r'\b(?:contact|speak|talk|reach|consult|ask)\b.{0,45}\b(?:clinician|professional|doctor|pharmacist|prescriber|therapist)\b',n)),
        'EMERGENCY_RESOURCE_BEHAVIOR':urgent,'CLINICAL_BOUNDARY_STATEMENT':boundary}
    return [r for r in requirements if not (flags[r] if r in flags else _required_visible_element_present(r,_normalized_for_match(text)))]


def evidence_support(text,evidence,claim_feature=False):
    """Exact lexical extract support only, preserving negation/modal/numbers.

    A paraphrase not verified by this narrow check is UNCERTAIN, not declared
    false. Unknown citations are checked separately; retrieved sources are not
    clinically approved by a matching excerpt.
    """
    clean=re.sub(r'\[source-\d+\]','',text)
    sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+|[;\n]+',clean) if s.strip()]
    claims=[]
    for sentence in sentences:
        # Only a whole refusal clause is exempt. A following promise or advice
        # sentence is assessed independently and cannot hide behind a disclaimer.
        if REFUSAL.search(sentence) and not re.search(r'\b(?:but|however|although|yet|and|because|you have)\b',sentence,re.I):continue
        if (MEDICAL.search(sentence) and not sentence.endswith('?')) or (claim_feature and re.search(r'\d|\b(?:cures?|causes?|proves?|guarantees?)\b',sentence,re.I)):
            claims.append(sentence)
    if not claims:return {'status':'UNCERTAIN' if claim_feature else 'NONE','claims':len(claims),'supporting_evidence_ids':[]}
    supported=[]
    for claim in claims:
        needle=normalized(claim)
        matches=[row['id'] for row in evidence if needle and needle in {
            normalized(sentence) for sentence in re.split(r'(?<=[.!?])\s+|\n+',row['text']) if sentence.strip()}]
        if not matches:return {'status':'UNVERIFIED','claims':len(claims),'supporting_evidence_ids':supported}
        supported.extend(matches[:1])
    return {'status':'EXTRACT_SUPPORTED','claims':len(claims),'supporting_evidence_ids':list(dict.fromkeys(supported))}


class BroadOutputModel:
    def __init__(self,root):
        import torch
        from transformers import AutoModelForSequenceClassification,AutoTokenizer
        root=Path(root)
        manifest=json.loads((root/'training_manifest.json').read_text(encoding='utf-8'))
        calibration=json.loads((root/'calibration.json').read_text(encoding='utf-8'))
        with (root/'model.safetensors').open('rb') as stream:checksum=hashlib.file_digest(stream,'sha256').hexdigest()
        if (manifest['model_sha256']!=checksum or manifest['calibration']!=calibration
                or calibration['labels']!=list(LABELS) or calibration['window_tokens']!=256
                or calibration['overlap_tokens']!=64 or calibration['max_input_tokens']!=4096
                or calibration['aggregation']!='per_label_max'
                or manifest['epochs_completed']<5):raise ValueError('Unpaired model/calibration contract')
        for key in ('thresholds','clear_thresholds'):
            if len(calibration[key])!=len(LABELS) or any(not 0<t<1 for t in calibration[key]):raise ValueError('Invalid thresholds')
        if any(a>=b for a,b in zip(calibration['clear_thresholds'],calibration['thresholds'])):raise ValueError('Invalid abstention band')
        if not torch.cuda.is_available():raise RuntimeError('CUDA required, no silent CPU fallback')
        self.torch=torch;self.calibration=calibration;self.model_sha256=checksum
        self.tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(root,local_files_only=True).float().cuda().eval()

    def assess(self,text):
        scores,tokens,windows=probabilities(self.model,self.tokenizer,text,self.torch)
        if scores is None:return {'status':'OUT_OF_SCOPE_LENGTH'}
        if len(scores)!=len(LABELS) or any(not math.isfinite(s) or not 0<=s<=1 for s in scores):
            return {'status':'INVALID_CLASSIFIER_SCORES'}
        return {'status':'ASSESSED','scores':dict(zip(LABELS,scores)),
            'positive':[c for c,s,t in zip(LABELS,scores,self.calibration['thresholds']) if s>=t],
            'uncertain':[c for c,s,lo,hi in zip(LABELS,scores,self.calibration['clear_thresholds'],self.calibration['thresholds']) if lo<=s<hi],
            'tokens':tokens,'windows':windows,'model_sha256':self.model_sha256}


class QwenFreeJudge:
    """No HTTP client, Qwen model, or generative judge is constructed here."""
    def __init__(self,model,checker):self.model,self.rules=model,checker

    @classmethod
    def load(cls,model_root,component_root):
        root=Path(component_root).resolve();sys.path.insert(0,str(root))
        import baseline
        Path(baseline.__file__).resolve().relative_to(root)
        from baseline.contracts import initialize_contracts
        from baseline.deterministic_checker import DeterministicChecker
        from rules.registry import RULE_REGISTRY
        from .local_output_rules import EXTRA_RULES
        initialize_contracts()
        return cls(BroadOutputModel(model_root),DeterministicChecker(rules=(*RULE_REGISTRY,*EXTRA_RULES),timeout_seconds=1.))

    def assess(self,value):
        tick=time.perf_counter();details={'guard_version':VERSION,'qwen_called':False,'clinical_deployment_ready':False}
        def finish(codes=(),error=None):
            return {'action':'ABSTAIN' if codes or error else 'DISPLAY','route':value.get('system_route') if type(value) is dict else None,
                'reason_codes':list(dict.fromkeys(codes)),'error_code':error,
                'diagnostics':{**details,'latency_ms':(time.perf_counter()-tick)*1000}}
        try:
            validate_input(value);text=value['candidate'];details['candidate_binding']=digest(value)
            rules=self.rules.check(text,{'system_route':value['system_route'],
                'response_policy':{**value['policy'],'required_behaviors':value['requirements']}})
            details['deterministic_status']=rules['status']
            codes=[r['reason_code'] for r in rules['violations']]
            if rules['status']!='OK':return finish(codes,'deterministic_check_failed')
            if codes:return finish(codes)
            requirements=list(value['requirements'])
            if value['system_route']=='crisis_response':
                requirements.extend(['CRISIS_SUPPORT','SAFETY_ORIENTED_GUIDANCE'])
            if value['system_route']=='urgent_medical_response' or value['policy'].get('severity')=='urgent':
                requirements.append('URGENT_ESCALATION')
            if value['system_route']=='unsupported_scope':requirements.append('scope_limit_notice')
            missing=requirement_missing(text,list(dict.fromkeys(requirements)));details['missing_requirements']=missing
            if missing:return finish(['POLICY_MISMATCH'])
            result=self.model.assess(text);details['classifier']=result
            if result['status']!='ASSESSED':return finish(error='classifier_out_of_scope')
            positive=set(result['positive']);uncertain=set(result['uncertain'])
            codes=[CODE_MAP.get(c,c) for c in positive-FEATURES]
            if 'ASSUMED_DISTRESS' in positive and value['policy'].get('interaction_kind')=='neutral_greeting':codes.append('UNSUPPORTED_DISTRESS_ASSUMPTION')
            if codes:return finish(codes)
            if uncertain-FEATURES:return finish(error='classifier_uncertain')
            if 'ASSUMED_DISTRESS' in uncertain and value['policy'].get('interaction_kind')=='neutral_greeting':return finish(error='greeting_context_uncertain')
            factual=evidence_support(text,value['evidence'],'MEDICAL_FACTUAL_CLAIM' in positive|uncertain);details['factuality']=factual
            if factual['status'] in {'UNCERTAIN','UNVERIFIED'}:return finish(['UNVERIFIED_MEDICAL_CLAIM'])
            cited=set(re.findall(r'\[(source-[^\]]+)\]',text))
            if cited and not cited<={row['id'] for row in value['evidence']}:
                # The active integration hashes evidence IDs. Without an
                # explicit marker map, do not pretend those hashes prove a
                # displayed source-N citation. This costs coverage on purpose.
                return finish(['UNVERIFIED_CITATION_BINDING'])
            return finish()
        except Exception:
            return finish(error='qwen_free_guard_failed')
