"""Generated text cannot invent or duplicate application-owned contact data.

This is a conservative contact-format check, not a crisis detector or a verified
resource directory. Ordinary numerical facts are not treated as phone numbers.
The caller must remove only the exact application-rendered resource prefix.
"""
import re

_SHORT_CONTACT=re.compile(r'(?<!\w)(?:9[ -]*8[ -]*8|9[ -]*1[ -]*1|1[ -]*1[ -]*2)(?!\w)')
_LONG_CONTACT=re.compile(r'(?<!\w)\+?\d[\d ().-]{5,}\d(?:[ -]*[A-Z]{3,})?(?!\w)',re.I)
_CONTACT_LINK=re.compile(r'(?:https?://|www\.)\S+',re.I)

def has_generated_contact_details(text):
    if _SHORT_CONTACT.search(text):return True
    for match in _LONG_CONTACT.finditer(text):
        value=match.group()
        if sum(c.isdigit() for c in value)>=7:return True
    if re.search(r'\b(?:helpline|hotline|lifeline|emergency|crisis service)\b',text,re.I):
        return bool(_CONTACT_LINK.search(text))
    return False
