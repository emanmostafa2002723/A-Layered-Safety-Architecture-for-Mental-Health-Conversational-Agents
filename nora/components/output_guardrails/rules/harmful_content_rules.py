from rules.base_rule import PatternRule

class HarmfulInstructionRule(PatternRule):
    rule_id = 'HARMFUL-INSTRUCTION'
    reason_code = 'HARMFUL_INSTRUCTIONS'
    severity = 'CRITICAL'
    patterns = (r'\b(?:you should|you can|try to|go ahead and)\s+(?:hurt|harm|kill|cut)\s+(?:yourself|someone|them)\b', r'^\s*(?:hurt|harm|kill|cut)\s+yourself\b')

class FalseReassuranceRule(PatternRule):
    rule_id = 'SAFETY-FALSE-REASSURANCE'
    reason_code = 'FALSE_REASSURANCE'
    ignore_negated = False
    patterns = (r'\byou (?:are|will be) (?:definitely|certainly|guaranteed) (?:safe|fine|okay|ok)\b', r'\byou (?:are definitely not|are not|aren\x27t) in (?:any )?danger\b', r'\bnothing (?:bad )?(?:can|will) happen to you\b')

class EmergencyContradictionRule(PatternRule):
    rule_id = 'EMERGENCY-CONTRADICTION'
    reason_code = 'EMERGENCY_GUIDANCE_CONTRADICTION'
    severity = 'CRITICAL'
    ignore_negated = False
    patterns = (r'\b(?:do not|don\x27t|never|avoid) (?:call|contact|seek)\b.{0,35}\b(?:emergency|crisis|ambulance|help)\b', r'\byou (?:do not|don\x27t) need\b.{0,30}\b(?:emergency|urgent|crisis)\b', r'\b(?:wait until tomorrow|delay (?:getting|seeking) help)\b')

    def check(self, response, context):
        if context.get('system_route') not in {'urgent_medical_response', 'crisis_response'}:
            return []
        return super().check(response, context)

class ProhibitedContentRule(PatternRule):
    rule_id = 'PROHIBITED-ABUSE'
    reason_code = 'PROHIBITED_CONTENT'
    patterns = (r'\byou (?:are|deserve to be) worthless\b', r'\byou deserve (?:to suffer|abuse)\b')
