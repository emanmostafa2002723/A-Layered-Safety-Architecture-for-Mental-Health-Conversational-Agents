"""Narrow conversational checks, not a crisis detector or safety authorization."""
import re


def unsupported_personal_memory(user_text, candidate, history, notes, packet):
    """Narrow unsupported-recall veto; no inference about the person's condition."""
    if packet.system_route.value != 'normal_chat' or packet.crisis_labels:
        return False
    if history:
        return False
    from .personal_memory import memory_recall_request
    candidate = candidate.replace('\u2019', "'")
    invented_advice = re.search(r"\b(?:i (?:previously |earlier )?(?:suggested|recommended|told you|asked you)|"
                               r"we (?:previously |earlier )?(?:discussed|talked about|agreed|worked on)|"
                               r"as (?:i|we) (?:discussed|mentioned|suggested|said))\b", candidate, re.I)
    invented_meeting = re.search(r'\b(?:our last|our previous|we last) (?:check.in|session|meeting|conversation|spoke)\b', candidate,re.I)
    if notes:
        return bool(invented_advice or invented_meeting)
    positive_recall = re.search(r"\b(?:i (?:remember|recall)|"
                               r"you (?:previously |earlier )?(?:mentioned|told me|said)|"
                               r"last time you|you've (?:mentioned|told me))\b", candidate, re.I)
    return bool(invented_advice or invented_meeting or (memory_recall_request(user_text) and positive_recall))


def ignored_listening_preference(user_text, candidate, notes, packet):
    """A narrow unsolicited-advice-list check on normal, no-crisis turns only."""
    if (packet.system_route.value != 'normal_chat' or packet.crisis_labels
            or not any(n.topic=='support_preference' and n.state=='listening' for n in notes)):
        return False
    if re.search(r'\b(?:what (?:can|should) i do|how (?:can|do|should) i|give me (?:advice|tips|ideas)|'
                 r'(?:want|need|please|some) (?:practical )?(?:advice|tips|suggestions|steps))\b',user_text,re.I):
        return False
    return bool(re.search(r'\b(?:a few things (?:that )?(?:might|can|could) help|'
                          r'first[, :] .+\b(?:second|then)\b)|(?:^|\n)\s*(?:1[.)]|[-*])\s+try\b',candidate,re.I|re.S))


def missed_available_memory(user_text, candidate, history, notes, packet):
    """A supplied summary is not verbatim recall, but denying it entirely is misleading."""
    if packet.system_route.value != 'normal_chat' or packet.crisis_labels or not notes or history:
        return False
    from .personal_memory import memory_recall_request
    return bool(memory_recall_request(user_text) and re.search(
        r"\bi (?:do not|don't|cannot|can't) (?:remember|recall) (?:what|anything)|"
        r"\bi (?:do not|don't) have (?:any |those )?(?:earlier |previous )?(?:details|memories|memory)\b",
        candidate.replace('\u2019', "'"), re.I))


def near_copied_reply(candidate,previous,packet):
    """Catch long copied paragraphs with a tiny tail added on normal turns."""
    if packet.system_route.value!='normal_chat' or packet.crisis_labels:return False
    normalize=lambda text:re.findall(r'[a-z0-9]+',text.casefold())
    current,old=normalize(candidate),normalize(previous)
    if min(len(current),len(old))<20:return False
    grams=lambda words:{tuple(words[i:i+5]) for i in range(len(words)-4)}
    a,b=grams(current),grams(old)
    # Both-direction coverage avoids flagging a short legitimate shared phrase.
    return len(a&b)/max(len(a),1)>=.80 and len(a&b)/max(len(b),1)>=.80


def reasked_answered_reason(user_text,candidate,history,packet):
    """Narrow normal-chat check; never suppress a repeated safety question."""
    if (packet.system_route.value!='normal_chat' or packet.crisis_tier.value!='normal'
            or packet.crisis_labels or packet.boundary_route.value!='allow'):
        return False
    previous=next((m['content'] for m in reversed(history) if m['role']=='assistant'),'')
    reason=re.compile(r"\b(?:what (?:is |has been )?mak(?:es|ing) you (?:feel )?(?:nervous|anxious|sad|worried|upset)|why (?:are you|do you feel) (?:nervous|anxious|sad|worried|upset))\b",re.I)
    answer=re.search(r"\b(?:because|it(?:'s| is) (?:the|my)|i (?:have|had|failed|faild|lost)|my (?:exam|test|interview|father|mother|partner|manager))\b",user_text,re.I)
    return bool(answer and reason.search(previous) and reason.search(candidate))


def ignored_topic_correction(user_text,candidate,packet):
    """Catch an explicit exam/interview correction, not broad topic inference."""
    if (packet.system_route.value!='normal_chat' or packet.crisis_tier.value!='normal'
            or packet.crisis_labels or packet.boundary_route.value!='allow'):
        return False
    topics={'exam':r'\b(?:exam|test|school)\b','interview':r'\binterview(?:ing|s)?\b'}
    denied=re.search(r'\bnot (?:a |an |the |my )?(exam|test|school|interview)\b',user_text,re.I)
    if denied is None:return False
    old='interview' if denied.group(1).casefold()=='interview' else 'exam'
    new='exam' if old=='interview' else 'interview'
    return bool(re.search(topics[new],user_text,re.I) and re.search(topics[old],candidate,re.I)
                and not re.search(topics[new],candidate,re.I))


def unsupported_greeting_distress(user_text, candidate, history, packet):
    """Do not infer distress from a fresh neutral greeting on a normal route.

    A greeting after a disclosure is NOT a fresh neutral greeting. The check
    never changes classifier labels, safety routes, or previous danger state.
    """
    if (packet.system_route.value != 'normal_chat'
            or packet.crisis_tier.value != 'normal'
            or packet.crisis_labels or packet.situation_codes
            or packet.boundary_route.value != 'allow'):
        return False
    greeting = re.compile(r'\s*(?:hi|hello|hey|good (?:morning|afternoon|evening))[!.,\s]*', re.I)
    if not greeting.fullmatch(user_text):
        return False
    if any(item['role'] == 'user' and not greeting.fullmatch(item['content']) for item in history):
        return False
    return bool(re.search(
        r"\b(?:sorry (?:that|to hear|you(?:'re| are))|"
        r"you(?:'re| are) (?:feeling this way|struggling|suffering|in pain|anxious|nervous|sad)|"
        r"I (?:can't|cannot) make (?:things|it) better|"
        r"suicid\w*|harm(?:ing)? yourself|hurt(?:ing)? yourself|in immediate danger|overwhelmed|"
        r"(?:difficult|hard|painful|terrible) (?:day|time|situation)|"
        r"what(?:'s| has| is) (?:been )?(?:so hard|happening lately that's been so hard))\b",
        candidate.replace('’', "'"), re.I))
