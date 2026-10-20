"""Narrow RAG-only recovery for explicit educational questions a router skips.

Never changes crisis/boundary decisions and never rewrites the original input.
Ambiguous fragments, support requests, roleplay and unrelated questions defer
to the router. This is not an intent or medical-safety classifier.
"""
import re


def clear_factual_mental_health_question(text):
    text=' '.join(text.casefold().replace('’',"'").split())
    conversational=re.search(r'\b(?:how are you|how about we|can we (?:just )?talk|(?:only|just) (?:want (?:someone )?to )?listen\w*|want someone to listen|story|roleplay|role play|poem|dialogue|fictional)\b',text)
    if conversational:return False
    # A medical word in a separate sentence is not the subject of a question.
    # "Coffee went cold. What is the weather?" must not trigger retrieval.
    for clause in re.split(r'[.!?;\n]+|\bbut\b|,(?=\s*(?:what|how|why|can|could)\b)',text):
        domain=re.search(r'\b(?:sleep\w*|insomnia|caffeine|coffee|panic|anxiet\w*|anxious|depress\w*|anhedonia|trauma|ptsd|cbt|cognitive behavioral therapy|ssri\w*)\b',clause)
        factual=re.search(r'\b(?:why|explain|define)\b|\bhow\b|\bwhat (?:is|are|does|do)\b',clause)
        factual=factual or re.search(r'\b(?:can|could|does|do)\s+(?:my\s+)?(?:coffee|caffeine|sleep|insomnia)\b.{0,100}\b(?:affect|cause|worsen|improve|disrupt|influence)\b',clause)
        if domain and factual:return True
    return False
