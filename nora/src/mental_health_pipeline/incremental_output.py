"""Experimental buffered-prefix veto, NOT a streaming approval guard.

Inspects complete clauses as decoding proceeds. Narrow literal assertions only;
negated/quoted/ambiguous clauses defer to the unchanged full-context final gate.
This is deliberately not an LLM call for every token and not general semantics.
"""
import re
import time


class IncrementalOutputStopped(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics=diagnostics
        super().__init__('incremental_output_veto')


def literal_clause_veto(text):
    for clause in re.findall(r'[^.!?\n]+[.!?\n]',text):
        # Conservative early veto: never decide from a negated/quoted fragment.
        if clause.rstrip().endswith('?'):
            continue
        if re.search(r"\b(?:not|never|cannot|can't|don't|doesn't|no|avoid|example|quote|said|says|say|told|reported|doctor|clinician|if)\b|[\"“”]",clause,re.I):
            continue
        if re.search(r'\byou (?:definitely |clearly |certainly )?(?:have|suffer from) (?:severe |major |clinical |an? )?(?:ADHD|depression|bipolar disorder|schizophrenia|anxiety disorder)\b',clause,re.I):
            return 'literal_personal_diagnosis'
        if re.search(r'\byou (?:should|must|need to) (?:start |increase |decrease |double )?(?:take|use) \d+(?:\.\d+)? (?:mg|milligrams?)\b',clause,re.I):
            return 'literal_personal_dose'
    return None


class BufferedPrefixVeto:
    """Transformers stopping-criterion callable; batch size one only.

All text remains internal. A veto raises after model.generate returns; its
truncated draft is never treated as a complete GenerationResult.
"""
    def __init__(self, tokenizer, prompt_length, *, interval=8):
        self.tokenizer=tokenizer; self.prompt_length=prompt_length; self.interval=interval
        self.checked_tokens=0; self.reason=None; self.checks=0; self.seconds=0.0
        self.stopped_tokens=0

    def __call__(self,input_ids,scores=None,**kwargs):
        count=len(input_ids[0])-self.prompt_length
        if self.reason: return True
        if count-self.checked_tokens < self.interval: return False
        self.checked_tokens=count
        started=time.perf_counter()
        text=self.tokenizer.decode(input_ids[0][self.prompt_length:],skip_special_tokens=True)
        self.reason=literal_clause_veto(text)
        self.checks+=1; self.seconds+=time.perf_counter()-started
        if self.reason: self.stopped_tokens=count
        return bool(self.reason)

    def diagnostics(self):
        return {'mode':'buffered_literal_clause_veto_v1','checks':self.checks,
                'checked_through_token':self.checked_tokens,'check_seconds':self.seconds,
                'aborted':bool(self.reason),'reason':self.reason,'stopped_tokens':self.stopped_tokens,
                'partial_text_released':False,'full_final_gate_required':True}
