"""Full Qwen/rules output checking, with checked medical-limit replacements.

No DeBERTa runtime or trained medical head is constructed by this backend.
Uncertainty, transport errors and mixed violations never authorize a notice.
"""
import hashlib

from .local_output import LocalQwenOutputGate

MEDICAL_CODES = frozenset({'PERSONALIZED_DIAGNOSIS','MEDICATION_RECOMMENDATION',
    'PERSONALIZED_MEDICATION_INSTRUCTION','INDIVIDUALIZED_TREATMENT_PLAN'})


class QwenOnlyOutputGate(LocalQwenOutputGate):
    def deliver(self, text, packet, retrieval=None):
        original=super().deliver(text,packet,retrieval)
        if original.get('action')=='DISPLAY': return original
        codes=original.get('reason_codes',[])
        if not codes or not set(codes)<=MEDICAL_CODES or original.get('error_code'):
            return original
        from .output_lab import replacement_for
        checked=super().deliver(replacement_for(codes),packet,retrieval)
        if checked.get('action')!='DISPLAY':
            return {**checked,'action':'ABSTAIN','text':None,'original_reason_codes':codes,
                    'replacement_checked':False}
        return {**checked,'action':'DISPLAY_CHECKED_REPLACEMENT','replacement_checked':True,
                'original_reason_codes':codes,'original_blocking_stage':'full_qwen_or_rules',
                'original_sha256':hashlib.sha256(text.encode()).hexdigest()}
