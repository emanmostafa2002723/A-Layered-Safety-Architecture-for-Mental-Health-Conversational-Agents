"""Narrow explicit request rules; safeguards are independent of medical intent.

No global 'I am safe', educational-keyword or profession exemption.  Every
clause is inspected so an innocent first clause cannot cancel a later request.
These rules establish policy intent, not a diagnosis or clinical assessment.
"""
from __future__ import annotations

import re
from .schema import normalize

MED = r"(?:medic(?:ine|ation)s?|drugs?|pills?|tablets?|antidepressants?|sertraline|fluoxetine|escitalopram|lithium|bupropion|clonazepam|methylphenidate|insulin|antibiotics?|prescri(?:bed|ption))"
CONDITION = r"(?:ADHD|OCD|PTSD|depression|bipolar(?: disorder)?|autism|cancer|anxiety disorder|schizophrenia|a disorder|a disease)"
DIRECT = [
    ("diagnosis", rf"\b(?:do i have|does (?:my|this|the) (?:partner|child|relative|patient|friend) have|am i (?:clinically )?(?:depressed|bipolar|autistic)|diagnose (?:me|my\b|this\b)|rule out {CONDITION} for (?:me|my|this)|confirm (?:that )?i have)\b"),
    ("diagnosis", r"\b(?:my (?:symptoms|screening score|test results?|lab results?).{0,100}(?:mean i have|prove|confirm|which disorder|what condition)|what does this (?:pattern|sound|look).{0,30}(?:in my case|about my diagnosis))\b"),
    ("medication_decision", rf"\b(?:what|which|how much|how many).{{0,30}}(?:dose|dosage|{MED}).{{0,40}}(?:should i|for me|for my|i should|to take|should (?:he|she|they|my))\b"),
    ("medication_decision", rf"\b(?:should i|can i|tell me how to|tell me whether to|choose|recommend|prescribe|decide whether i should).{{0,40}}(?:start|stop|switch|double|increase|reduce|combine|take|dose|{MED}).{{0,55}}\b{MED}\b"),
    ("medication_decision", rf"\b(?:prescribe|choose a dose|create a dosing schedule|pick (?:a|an) (?:antidepressant|medicine)|double my antidepressant|how to stop my prescribed medication)\b"),
    ("clinical_treatment", r"\b(?:personalized|personalised|individualized|individualised)\s+(?:clinical\s+)?treatment plan\b"),
    ("clinical_treatment", r"\b(?:replace my (?:therapist|psychiatrist)|act as my (?:therapist|psychiatrist)|decide (?:whether|which).{0,60}(?:therapy|treatment).{0,45}(?:for me|my condition))\b"),
    ("harmful_medical_instruction", r"\b(?:how (?:do|can|should) i|give me instructions to|tell me how to|help me).{0,55}(?:secretly sedate|poison|falsify medical|forge a prescription|hide.{0,15}overdose|operate on.{0,30}without|sedate.{0,40}without.{0,15}consent)\b"),
]
DIRECT = [(name, re.compile(pattern, re.I)) for name, pattern in DIRECT]
CLAUSES = re.compile(r"[.!?;\n]+|\b(?:but|however|and now)\b", re.I)
NEGATED_EVENT = re.compile(r"\b(?:did not|didn't|have not|haven't|has not|hasn't|not|never|no one)\s+(?:\w+\s+){0,3}$", re.I)
HISTORY = re.compile(r"\b(?:years? ago|months? ago|last year|history of|used to)\b", re.I)
PERSON = re.compile(r"\b(?:i|i'm|my|our|he|she|they|someone|friend|partner|neighbor|neighbour|child|patient|person)\b", re.I)
EDUCATIONAL = re.compile(r"^\s*(?:what (?:is|are)|explain|define|for (?:a class|an essay|general education)|in general)\b", re.I)
SIGNS = re.compile(r"\b(?:unconscious|unresponsive|won't wake up|cannot breathe|can't breathe|throat (?:is )?closing|having a seizure|keep fainting|severe allergic reaction)\b", re.I)
INGESTION = re.compile(rf"\b(?:overdos(?:ed|ing)|took (?:far )?too (?:much|many)|swallowed (?:several|too many|a handful)|taken (?:far )?more).{{0,60}}(?:{MED})?\b", re.I)
CHILD = re.compile(r"\b(?:child|kid|son|daughter|baby|\w+(?:-| )years?(?:-| )old|eight-year-old|ten-year-old)\b", re.I)
ABUSE = re.compile(r"\b(?:being (?:beaten|hit|abused|hurt)|(?:beats|hits|is beating|is hitting|hurts) (?:him|her|them|me|my)|is afraid to go home)\b", re.I)
SUICIDE = re.compile(r"\b(?:wants? to (?:end (?:his|her|their|my) life|kill (?:himself|herself|themselves|myself))|going to kill (?:himself|herself|themselves|myself))\b", re.I)
STRICT_EDIT = re.compile(r'''^\s*(?:please )?(?:proofread|translate|rewrite) (?:this|the following)(?: sentence| text| question)? without answering it:\s*(?:"[^"\n]+"|'[^'\n]+')\s*$''', re.I)
DIAGNOSIS_EVIDENCE = re.compile(r"\b(?:diagnos\w*|symptoms?|disorders?|diseases?|screening|test results?|lab results?|clinically|what is wrong with|what's wrong with|sound like|pattern|ADHD|OCD|PTSD|bipolar|autistic|depressed|depression|autism|schizophrenia)\b", re.I)


def nli_eligibility(text):
    """Interim NLI scope checks, never gates on age/residence or pronouns alone.

    Diagnosis requires some evidence about assessing a condition. Reporting
    abuse or asking for help alone cannot establish that request. Supervised
    inference is evaluated independently and does not use this lexical gate.
    """
    return {"diagnosis":bool(DIAGNOSIS_EVIDENCE.search(text)),
            "medication_decision":True,"clinical_treatment":True,
            "harmful_medical_instruction":True}


def inspect(text: str) -> dict:
    text = normalize(text)
    if STRICT_EDIT.fullmatch(text):
        return {"reasons": [], "safety_signals": [], "rule_ids": ["quoted_transformation_only"]}
    reasons, signals, ids = set(), set(), []
    for index, (name, pattern) in enumerate(DIRECT):
        for match in pattern.finditer(text):
            prefix = text[max(0, match.start()-35):match.start()]
            # Limited negation of an instruction, not a whole-message bypass.
            if re.search(r"\b(?:do not|don't|not asking you to)\s*$", prefix, re.I):
                continue
            reasons.add(name)
            ids.append(f"request_{index}")
    clauses = [c.strip() for c in CLAUSES.split(text) if c.strip()]
    # Child identity may precede an abuse clause in an adjacent sentence.
    for index, clause in enumerate(clauses):
        if HISTORY.search(clause) or EDUCATIONAL.search(clause):
            continue
        for event in list(SIGNS.finditer(clause)) + list(INGESTION.finditer(clause)):
            if NEGATED_EVENT.search(clause[:event.start()]):
                continue
            if PERSON.search(clause):
                # 'too much work' alone must never match ingestion.
                if SIGNS.fullmatch(event.group()) or re.search(r"overdos|swallowed|taken|took", event.group(), re.I) and re.search(rf"\b{MED}\b", clause, re.I):
                    signals.add("medical_emergency")
        previous = clauses[index-1] if index else ""
        abuse = ABUSE.search(clause)
        if abuse and CHILD.search(previous + " " + clause) and not NEGATED_EVENT.search(clause[:abuse.start()]):
            signals.add("child_safeguarding")
        suicide = SUICIDE.search(clause)
        if suicide and not NEGATED_EVENT.search(clause[:suicide.start()]):
            signals.add("suicide_safety")
    return {"reasons": sorted(reasons), "safety_signals": sorted(signals), "rule_ids": sorted(set(ids))}

