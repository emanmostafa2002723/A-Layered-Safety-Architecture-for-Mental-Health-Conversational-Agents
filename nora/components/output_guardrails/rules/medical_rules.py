"""Explicit English patterns. Semantic detection remains separately required."""
from rules.base_rule import PatternRule
MEDICATION = r'(?:sertraline|fluoxetine|escitalopram|citalopram|paroxetine|venlafaxine|duloxetine|lithium|alprazolam|lorazepam|diazepam|clonazepam|antidepressants?|medication|medicine|meds|pills?|dose|dosage)'
PERSONAL = r'(?:you (?:should|must|need to|can)|i (?:recommend|suggest)(?: that)? you)\s+'
DIRECT = r'(?:\b' + PERSONAL + r'|^\s*|\byour (?:doctor|clinician) (?:said|told you) to\s+)'

class PersonalizedDiagnosisRule(PatternRule):
    rule_id = 'DG-PERSONALIZED-DIAGNOSIS'
    reason_code = 'PERSONALIZED_DIAGNOSIS'
    patterns = (
        r'\byou (?:definitely |clearly |certainly )?(?:have|suffer from|are suffering from) (?:clinical )?(?:depression|anxiety(?: disorder)?|bipolar(?: disorder)?|ptsd|schizophrenia|ocd|adhd)\b',
        r'\byou are (?:clinically )?(?:depressed|bipolar|schizophrenic)\b', r'\bi diagnose you with\b',
    )

class MedicationRecommendationRule(PatternRule):
    rule_id = 'MED-PERSONALIZED-RECOMMENDATION'
    reason_code = 'MEDICATION_RECOMMENDATION'
    patterns = (DIRECT + r'(?:take|try|use|start|begin)\s+(?:taking\s+)?(?:some |a |an |your )?' + MEDICATION + r'\b',)

class MedicationDoseRule(PatternRule):
    rule_id = 'MED-DOSE'
    reason_code = 'PERSONALIZED_MEDICATION_INSTRUCTION'
    severity = 'CRITICAL'
    patterns = (DIRECT + r'(?:take|use|increase|decrease|raise|lower|change)\b.{0,70}?\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|micrograms?|milligrams?|ml|tablets?|pills?)\b',
        r'\b(?:double|halve|triple) (?:your|the) (?:dose|medication|pills?)\b')

class MedicationScheduleRule(PatternRule):
    rule_id = 'MED-SCHEDULE'
    reason_code = 'PERSONALIZED_MEDICATION_INSTRUCTION'
    severity = 'CRITICAL'
    patterns = (DIRECT + r'take\b.{0,50}?' + MEDICATION + r'.{0,40}?(?:every \d+ hours|twice (?:a day|daily)|each (?:morning|night)|tonight)\b',)

class MedicationStartRule(PatternRule):
    rule_id = 'MED-START'
    reason_code = 'PERSONALIZED_MEDICATION_INSTRUCTION'
    severity = 'CRITICAL'
    patterns = (DIRECT + r'(?:start|begin)\s+(?:taking\s+)?(?:your |the )?' + MEDICATION + r'\b',)

class MedicationStopRule(PatternRule):
    rule_id = 'MED-STOP'
    reason_code = 'PERSONALIZED_MEDICATION_INSTRUCTION'
    severity = 'CRITICAL'
    patterns = (r'\b(?:stop|discontinue|quit)\s+(?:taking\s+)?(?:your |the )?' + MEDICATION + r'\b',)

class MedicationSwitchRule(PatternRule):
    rule_id = 'MED-SWITCH'
    reason_code = 'PERSONALIZED_MEDICATION_INSTRUCTION'
    severity = 'CRITICAL'
    patterns = (r'\b(?:switch|change)\s+(?:from\s+)?(?:your |the )?' + MEDICATION + r'\b',)

class IndividualizedTreatmentRule(PatternRule):
    rule_id = 'TREATMENT-INDIVIDUALIZED'
    reason_code = 'INDIVIDUALIZED_TREATMENT_PLAN'
    patterns = (r'\byour treatment plan (?:is|will be|should be)\b', PERSONAL + r'(?:do|undergo|begin)\b.{0,30}\b(?:therapy|treatment)\b')

class SymptomInterpretationRule(PatternRule):
    rule_id = 'CLINICAL-INTERPRETATION'
    reason_code = 'PERSONAL_CLINICAL_INTERPRETATION'
    patterns = (r'\byour symptoms (?:mean|prove|confirm|show)\b', r'\bbased on your symptoms,? you (?:have|are)\b')
