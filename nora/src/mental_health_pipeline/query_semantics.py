"""Conservative search-only context/rewrite checks, never safety routing.

Original user text is preserved elsewhere. Ambiguous references stay unresolved;
no facts are extracted from assistant claims or generated summaries.
"""
import re

SUBJECTS={
    'cognitive behavioral therapy':r'\b(?:CBT|cognitiv\w* behav\w* ther(?:ap|p)\w*)\b',
    'dialectical behavior therapy':r'\b(?:DBT|dialectical behavio\w* therap\w*)\b',
    'EMDR':r'\bEMDR\b',
    'exposure therapy':r'\bexposure therapy\b',
    'antidepressants':r'\bantidepressants?\b',
    'sleep hygiene':r'\bsleep hygiene\b',
}
ANCHORS={**SUBJECTS,
    'caffeine':r'\b(?:cof+e+e?|caffeine|caffiene)\b',
    'sleep':r'\b(?:sleep\w*|insomnia)\b',
    'worry':r'\b(?:worr\w*|anxi\w*)\b',
    'shift':r'\b(?:shifts?|shiftwork|night work)\b',
    'panic':r'\bpanic\b','trauma':r'\b(?:trauma\w*|flashbacks?)\b','PTSD':r'\bPTSD\b',
    'medication':r'\b(?:medicin\w*|medication\w*|tablets?|pills?|dosage|doses?)\b',
}


def _factual_followup(latest):
    if not re.search(r'^\s*(?:and (?:for|what|how)|what about|how about|(?:does|can|will|would|is) (?:it|that)\b)',latest,re.I):
        return False
    if re.search(r'\b(?:listen|just talk|weather|repeat|say again)\b',latest,re.I):return False
    if not re.search(r'\b(?:help|work|effective|safe|risks?|side effects?|benefits?|cost|evidence|anxi\w*|worr\w*|depress\w*|sleep\w*|trauma|panic)\b',latest,re.I):
        return False
    return True


def contextual_search_query(latest,history):
    """Resolve explicit subjects for factual followups; never invent a subject."""
    if any(re.search(p,latest,re.I) for p in SUBJECTS.values()):return latest,False
    if not _factual_followup(latest):return latest,False
    found=[]
    # Carry an explicit user subject through at most four uninterrupted factual
    # followups, never through a topic change or an assistant-generated claim.
    previous_users=[m['content'] for m in history if m['role']=='user'][-4:]
    for previous in reversed(previous_users):
        found=[name for name,p in SUBJECTS.items() if re.search(p,previous,re.I)]
        if found:break
        if not _factual_followup(previous):break
    if len(found)!=1:return latest,False
    if re.search(r'\b(?:it|that)\b',latest,re.I):
        return re.sub(r'\b(?:it|that)\b',found[0],latest,count=1,flags=re.I),True
    return latest+' (about '+found[0]+')',True


def preserve_query_meaning(original,canonical):
    """Restore observable subject loss; passing this is NOT semantic proof."""
    missing=[name for name,p in ANCHORS.items() if re.search(p,original,re.I) and not re.search(p,canonical,re.I)]
    reasons=['lost_explicit_subject:'+name for name in missing]
    clinical=r'(?:sleep\w*|anx\w*|depress\w*|medicin\w*|medication\w*|therapy|treatment|suicid\w*|harm\w*)'
    negative=r"\b(?:not|never|without|no|don.t|cannot|can.t)\b(?:\W+\w+){0,4}\W+"+clinical
    if re.search(negative,original,re.I) and not re.search(r"\b(?:not|never|without|no|don.t|cannot|can.t)\b",canonical,re.I):
        reasons.append('clinical_negation_not_preserved')
    return (original,tuple(reasons)) if reasons else (canonical,())


def evidence_query_terms(text):
    """Remove question glue and expand explicit aliases for evidence ranking."""
    for name,p in SUBJECTS.items():
        if re.search(p,text,re.I):text+=' '+name
    stop={'a','an','the','what','is','are','how','does','do','can','could','would','will','it','that',
          'for','to','of','and','or','with','about','in','on','i','my','me','you','your','someone','please','explain','help'}
    return set(re.findall(r'\w+',text.casefold()))-stop


def subject_aligned_documents(query,documents):
    """Do not substitute other treatments or clearly unrequested title scopes.

    Lexical relevance only, not source approval or evidence of a medical claim.
    Leave multi-method comparisons alone; never invent missing evidence.
    """
    explicit=[name for name,p in SUBJECTS.items() if re.search(p,query,re.I)]
    methods=[name for name in explicit if name not in {'antidepressants','sleep hygiene'}]
    if len(explicit)!=1 or len(methods)!=1:return list(documents)
    pattern=SUBJECTS[methods[0]]
    scopes=(r'\b(?:BPD|borderline personality)\b',r'\bbipolar\b',r'\bschizophren\w*\b',
            r'\b(?:PTSD|post.traumatic)\b',r'\bpanic\b',r'\b(?:OCD|obsessive.compulsive)\b')
    selected=[]
    for doc in documents:
        if not re.search(pattern,doc.text,re.I):continue
        if any(re.search(p,doc.citation.title or '',re.I) and not re.search(p,query,re.I) for p in scopes):continue
        selected.append(doc)
    return selected
