from rules.medical_rules import (PersonalizedDiagnosisRule, MedicationRecommendationRule, MedicationDoseRule,
    MedicationScheduleRule, MedicationStartRule, MedicationStopRule, MedicationSwitchRule,
    IndividualizedTreatmentRule, SymptomInterpretationRule)
from rules.harmful_content_rules import (HarmfulInstructionRule, FalseReassuranceRule,
    EmergencyContradictionRule, ProhibitedContentRule)
from rules.privacy_rules import PrivacyLeakRule
from rules.leakage_rules import SystemPromptLeakRule
RULE_REGISTRY = (PersonalizedDiagnosisRule(), MedicationRecommendationRule(), MedicationDoseRule(),
    MedicationScheduleRule(), MedicationStartRule(), MedicationStopRule(), MedicationSwitchRule(),
    IndividualizedTreatmentRule(), SymptomInterpretationRule(), HarmfulInstructionRule(),
    FalseReassuranceRule(), EmergencyContradictionRule(), PrivacyLeakRule(), SystemPromptLeakRule(), ProhibitedContentRule())
