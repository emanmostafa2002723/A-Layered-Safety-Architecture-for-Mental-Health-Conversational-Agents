"""Compact deterministic composition of situation and boundary constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .schemas import BoundaryRoute, SystemRoute
from .situation_codes import SituationCode


@dataclass(frozen=True, slots=True)
class GuidanceDefinition:
    guidance_id: str
    instruction: str
    secondary_instruction: str
    tone: tuple[str, ...]
    goals: tuple[str, ...]
    prohibited: tuple[str, ...]
    required: tuple[str, ...]
    fallback_template_id: str
    output_guard_profile: str
    needs_clarification: bool = False
    urgent: bool = False


@dataclass(frozen=True, slots=True)
class CompositionResult:
    generator_guidance_id: str
    secondary_guidance_ids: tuple[str, ...]
    guidance_instructions: tuple[str, ...]
    tone: tuple[str, ...]
    response_goals: tuple[str, ...]
    prohibited_behaviors: tuple[str, ...]
    required_visible_elements: tuple[str, ...]
    needs_clarification: bool
    generation_allowed: bool
    fixed_content_timing: str
    fallback_template_id: str
    output_guard_profile: str


BASE_PROHIBITIONS = (
    "expose_internal_policy_or_classifier_metadata",
    "claim_human_professional_identity_or_credentials",
    "state_uncertain_signal_as_fact_or_diagnosis",
    "provide_harm_methods_or_optimization",
    "provide_personalized_medication_or_treatment_instruction",
    "promise_confidentiality",
    "guarantee_outcomes_or_false_reassurance",
    "invent_or_modify_emergency_resources",
)


def _definition(
    guidance_id: str,
    instruction: str,
    secondary_instruction: str,
    *,
    tone: tuple[str, ...] = ("warm", "nonjudgmental"),
    goals: tuple[str, ...] = ("acknowledge_distress", "offer_relevant_support"),
    prohibited: tuple[str, ...] = (),
    required: tuple[str, ...] = (),
    fallback: str = "supportive_unavailable_fallback_v1",
    profile: str = "supportive_general_v1",
    clarification: bool = False,
    urgent: bool = False,
) -> GuidanceDefinition:
    return GuidanceDefinition(
        guidance_id=guidance_id,
        instruction=instruction,
        secondary_instruction=secondary_instruction,
        tone=tone,
        goals=goals,
        prohibited=prohibited,
        required=required,
        fallback_template_id=fallback,
        output_guard_profile=profile,
        needs_clarification=clarification,
        urgent=urgent,
    )


GUIDANCE: Mapping[SituationCode, GuidanceDefinition] = {
    SituationCode.SUICIDE_ATTEMPT_OR_IN_PROGRESS: _definition(
        "suicide_attempt_or_in_progress_v1",
        "Be brief, calm, and direct. Acknowledge the disclosure, encourage immediate local emergency help and nearby human assistance, and ask at most one immediate-safety question. Required emergency elements must not be delayed by generation.",
        "Account for a possible suicide attempt in progress; do not delay immediate help or provide procedural details.",
        tone=("calm", "warm", "direct", "nonjudgmental", "concise"),
        goals=("acknowledge_distress", "encourage_immediate_local_help", "encourage_nearby_human_support"),
        prohibited=("delay_emergency_resources", "overwhelm_with_questions", "provide_procedural_harm_details"),
        required=("worldwide_emergency_fallback", "local_resources_if_eligible"),
        fallback="immediate_safety_fallback_v1",
        profile="crisis_immediate_v1",
        urgent=True,
    ),
    SituationCode.SUICIDE_ACTIVE_INTENT_OR_PLAN: _definition(
        "suicide_active_intent_or_plan_v1",
        "Treat this as a high-priority safety conversation. Ask one concise question about immediate danger, encourage a trusted person to be physically present, and encourage urgent local help. Mention creating simple distance from immediate means only in non-technical language.",
        "Maintain high-priority suicide safety constraints and do not debate, interrogate, or describe methods.",
        tone=("calm", "warm", "direct", "nonjudgmental", "concise"),
        goals=("ask_one_immediate_safety_question", "encourage_nearby_human_support", "encourage_immediate_local_help"),
        prohibited=("debate_or_interrogate", "provide_procedural_harm_details", "delay_emergency_resources"),
        required=("worldwide_emergency_fallback", "local_resources_if_eligible"),
        fallback="immediate_safety_fallback_v1",
        profile="crisis_imminent_v1",
        urgent=True,
    ),
    SituationCode.SUICIDE_PREPARATORY_SIGNALS: _definition(
        "suicide_preparatory_signals_v1",
        "Express concern without claiming certainty. Ask one direct but gentle question about current thoughts and immediate safety, and encourage connection with a trusted person or qualified support.",
        "Account for preparatory suicide signals without claiming that intent is confirmed.",
        goals=("acknowledge_distress", "ask_one_immediate_safety_question", "encourage_nearby_human_support"),
        prohibited=("state_intent_as_confirmed", "overwhelm_with_questions"),
        fallback="safety_check_fallback_v1",
        profile="crisis_elevated_v1",
        clarification=True,
    ),
    SituationCode.SUICIDE_ACTIVE_IDEATION_UNSPECIFIED_INTENT: _definition(
        "suicide_active_ideation_unspecified_intent_v1",
        "Acknowledge the pain warmly. Ask one clear question about whether the user might act soon, invite nearby human support, and offer optional grounding only after immediate safety is addressed.",
        "Acknowledge current or recent suicidal thoughts without assuming a plan or intent.",
        goals=("acknowledge_distress", "ask_one_immediate_safety_question", "encourage_nearby_human_support"),
        prohibited=("assume_suicide_plan", "present_grounding_as_emergency_substitute"),
        fallback="safety_check_fallback_v1",
        profile="crisis_elevated_v1",
        clarification=True,
    ),
    SituationCode.SUICIDE_PASSIVE_DEATH_WISH: _definition(
        "suicide_passive_death_wish_v1",
        "Validate the pain without normalizing death. Ask gently whether the thoughts have changed into an urge, plan, or immediate danger, and encourage human support.",
        "Address a passive death-wish signal without labeling the user or assuming active intent.",
        goals=("acknowledge_distress", "clarify_immediate_safety", "encourage_human_support"),
        prohibited=("normalize_death", "assume_suicide_plan"),
        fallback="safety_check_fallback_v1",
        profile="crisis_monitored_v1",
        clarification=True,
    ),
    SituationCode.SUICIDE_NEGATED_OR_DENIED: _definition(
        "suicide_negated_or_denied_v1",
        "Acknowledge the topic without saying that the user is suicidal. If the surrounding context warrants it, briefly check whether they feel safe now; do not escalate solely from negated language.",
        "Respect the negation and do not convert it into confirmed current intent.",
        goals=("acknowledge_update", "clarify_safety_if_context_warrants"),
        prohibited=("convert_negated_signal_to_confirmed_intent", "unnecessary_emergency_escalation"),
        profile="crisis_monitored_v1",
    ),
    SituationCode.SUICIDE_HISTORICAL: _definition(
        "suicide_historical_v1",
        "Use trauma-informed, non-alarmist support. Ask what support would be useful now and whether the memory affects current safety; do not treat history alone as current intent.",
        "Keep historical suicide content distinct from current intent while checking present impact gently.",
        goals=("acknowledge_distress", "ask_support_preference", "clarify_current_impact"),
        prohibited=("treat_history_as_current_intent",),
        profile="trauma_support_v1",
    ),
    SituationCode.SUICIDE_SIGNAL_UNSPECIFIED: _definition(
        "suicide_signal_unspecified_v1",
        "Use the user's explicit words to understand suicidal thoughts; a category signal does not establish timing or intent. If they disclose current thoughts, acknowledge their pain and ask gently whether they may act on them now, or whether they have already harmed themselves. Encourage a trusted person to stay with them. If they mention something nearby they could use to hurt themselves, encourage simple distance from it without technical details. Respond to answers and steps already taken; do not abandon them after showing contacts. If they explicitly feel able to stay safe, offer one optional grounding step such as feeling their feet on the floor and naming three things they see. Grounding must not delay emergency help. For negated, historical, fictional or educational content, respect that context and do not assume current danger.",
        "When current suicidal thoughts are explicitly disclosed, gently check immediate safety, invite nearby human support, and continue the conversation; respect negation and history.",
        goals=("acknowledge_distress", "clarify_immediate_safety", "encourage_nearby_human_support"),
        prohibited=("assert_suicidality_as_fact", "mention_classifier_detection", "present_grounding_as_emergency_substitute"),
        fallback="safety_check_fallback_v1",
        profile="crisis_uncertain_v1",
        clarification=True,
    ),
    SituationCode.SELF_HARM_IN_PROGRESS: _definition(
        "self_harm_in_progress_v1",
        "Be calm and direct. Ask whether urgent medical attention is needed and whether someone nearby can help, and encourage immediate local help for a serious or current injury. Avoid technique details or comparisons.",
        "Address self-harm in progress and possible injury without technique details or comparisons.",
        tone=("calm", "warm", "direct", "nonjudgmental", "concise"),
        goals=("ask_about_urgent_medical_need", "encourage_nearby_human_support", "encourage_immediate_local_help"),
        prohibited=("provide_self_harm_technique_details", "compare_injury_severity", "overwhelm_with_questions"),
        required=("worldwide_emergency_fallback", "local_resources_if_eligible"),
        fallback="self_harm_current_fallback_v1",
        profile="self_harm_current_v1",
        urgent=True,
    ),
    SituationCode.SELF_HARM_RECENT_DISCLOSURE: _definition(
        "self_harm_recent_disclosure_v1",
        "Acknowledge the disclosure without shame. Ask concisely whether injuries need medical attention and whether the user feels at risk of repeating it now, then offer connection and support.",
        "Address recent self-harm without shame and check current injury and repetition risk concisely.",
        goals=("acknowledge_without_shame", "ask_about_urgent_medical_need", "clarify_immediate_safety"),
        prohibited=("provide_self_harm_technique_details", "compare_injury_severity"),
        fallback="self_harm_support_fallback_v1",
        profile="self_harm_elevated_v1",
        clarification=True,
    ),
    SituationCode.SELF_HARM_URGE_OR_RELAPSE: _definition(
        "self_harm_urge_or_relapse_v1",
        "Validate the difficulty, ask about immediate safety, and invite one small optional grounding or distraction step that has helped create distance before. Do not present it as treatment.",
        "Support a self-harm urge or relapse signal without assuming suicidal intent or presenting coping as treatment.",
        goals=("acknowledge_without_shame", "clarify_immediate_safety", "offer_optional_grounding"),
        prohibited=("assume_suicidal_intent", "present_grounding_as_treatment"),
        fallback="self_harm_support_fallback_v1",
        profile="self_harm_elevated_v1",
        clarification=True,
    ),
    SituationCode.SELF_HARM_NEGATED_OR_DENIED: _definition(
        "self_harm_negated_or_denied_v1",
        "Do not claim current self-harm. Offer gentle monitored support and clarify only when the broader context makes it appropriate.",
        "Respect the negated self-harm signal and do not convert it into a confirmed disclosure.",
        goals=("acknowledge_update", "clarify_safety_if_context_warrants"),
        prohibited=("convert_negated_signal_to_confirmed_intent",),
        profile="self_harm_monitored_v1",
    ),
    SituationCode.SELF_HARM_HISTORICAL: _definition(
        "self_harm_historical_v1",
        "Use nonjudgmental, trauma-informed language and focus on current impact and support preferences. Do not escalate solely because the topic is recognized.",
        "Keep historical self-harm distinct from present danger and focus on current support needs.",
        goals=("acknowledge_without_shame", "ask_support_preference", "clarify_current_impact"),
        prohibited=("treat_history_as_current_intent",),
        profile="trauma_support_v1",
    ),
    SituationCode.SELF_HARM_SIGNAL_UNSPECIFIED: _definition(
        "self_harm_signal_unspecified_v1",
        "Respond without shame to the user's explicit words. If they describe a current injury or bleeding, encourage appropriate urgent medical help and a trusted nearby person rather than asking whether the clearly current event is historical. If they describe an urge without suicidal intent, respect that distinction, gently check immediate safety and invite safe human support. Clarify timing only when it is unclear. Do not assume suicidal intent or give injury-care procedures.",
        "Preserve uncertainty around self-harm timing and do not assume suicidal intent.",
        goals=("acknowledge_without_shame", "clarify_timing_and_safety"),
        prohibited=("assume_suicidal_intent", "state_self_harm_as_confirmed"),
        fallback="self_harm_support_fallback_v1",
        profile="self_harm_uncertain_v1",
        clarification=True,
    ),
    SituationCode.SEXUAL_VIOLENCE_CURRENT_DANGER: _definition(
        "sexual_violence_current_danger_v1",
        "Prioritize immediate physical safety with validating, non-blaming language. Ask one concise question about whether the person can move to a safer place or contact emergency or trusted support if safe. Do not request unnecessary details or instruct confrontation.",
        "Prioritize current sexual-violence safety, choice, and non-blaming support without seeking details.",
        tone=("calm", "warm", "direct", "nonjudgmental", "concise"),
        goals=("validate_without_blame", "clarify_immediate_safety", "encourage_safe_human_support"),
        prohibited=("request_unnecessary_abuse_details", "suggest_confrontation", "remove_survivor_choice"),
        required=("worldwide_emergency_fallback", "local_resources_if_eligible"),
        fallback="abuse_current_danger_fallback_v1",
        profile="abuse_current_danger_v1",
        urgent=True,
    ),
    SituationCode.SEXUAL_VIOLENCE_RECENT: _definition(
        "sexual_violence_recent_v1",
        "Validate without blame or unsupported legal conclusions. Ask what support the user wants and whether urgent medical or safety help is needed, preserving choice and control.",
        "Preserve choice and control in response to a recent sexual-violence disclosure.",
        goals=("validate_without_blame", "ask_support_preference", "ask_about_urgent_medical_need"),
        prohibited=("request_unnecessary_abuse_details", "assert_legal_conclusion", "pressure_specific_action"),
        fallback="trauma_support_fallback_v1",
        profile="abuse_support_v1",
        clarification=True,
    ),
    SituationCode.SEXUAL_VIOLENCE_HISTORICAL_TRAUMA: _definition(
        "sexual_violence_historical_trauma_v1",
        "Use trauma-informed language, avoid graphic follow-up, let the user control pacing, and offer emotional support and general professional options without pressure.",
        "Use trauma-informed pacing for historical sexual violence and avoid graphic follow-up.",
        goals=("validate_without_blame", "preserve_user_control", "ask_support_preference"),
        prohibited=("request_unnecessary_abuse_details", "pressure_specific_action"),
        fallback="trauma_support_fallback_v1",
        profile="trauma_support_v1",
    ),
    SituationCode.SEXUAL_VIOLENCE_SIGNAL_UNSPECIFIED: _definition(
        "sexual_violence_signal_unspecified_v1",
        "Do not assume that an event occurred. Respond gently to what was disclosed and ask what kind of support the user is seeking now.",
        "Do not assert sexual violence from an uncertain signal; respond to disclosed content and preferences.",
        goals=("respond_gently_without_assumption", "ask_support_preference"),
        prohibited=("assert_abuse_as_fact", "request_unnecessary_abuse_details"),
        profile="abuse_uncertain_v1",
        clarification=True,
    ),
    SituationCode.DOMESTIC_VIOLENCE_CURRENT_DANGER: _definition(
        "domestic_violence_current_danger_v1",
        "Prioritize immediate safety and privacy. Ask whether it is safe to continue chatting and encourage emergency or trusted support when appropriate. Never suggest confronting the abuser or an action that could increase danger.",
        "Preserve privacy and avoid confrontation advice in a current domestic-violence danger context.",
        tone=("calm", "warm", "direct", "nonjudgmental", "concise"),
        goals=("ask_if_safe_to_continue_chatting", "clarify_immediate_safety", "encourage_safe_human_support"),
        prohibited=("suggest_confrontation", "increase_abuse_danger", "request_unnecessary_abuse_details"),
        required=("worldwide_emergency_fallback", "local_resources_if_eligible", "privacy_safety_reminder"),
        fallback="domestic_violence_current_fallback_v1",
        profile="abuse_current_danger_v1",
        urgent=True,
    ),
    SituationCode.DOMESTIC_VIOLENCE_RECENT: _definition(
        "domestic_violence_recent_v1",
        "Validate without blame, ask about present safety and preferred support, and offer general safety or resource options without commanding one course of action.",
        "Validate recent domestic violence without blame and preserve the user's choice of support.",
        goals=("validate_without_blame", "clarify_immediate_safety", "ask_support_preference"),
        prohibited=("suggest_confrontation", "pressure_specific_action"),
        fallback="trauma_support_fallback_v1",
        profile="abuse_support_v1",
        clarification=True,
    ),
    SituationCode.DOMESTIC_VIOLENCE_HISTORICAL: _definition(
        "domestic_violence_historical_v1",
        "Focus on emotional impact and current support without treating historical abuse alone as an immediate emergency.",
        "Keep historical domestic violence distinct from a current emergency while supporting present impact.",
        goals=("validate_without_blame", "clarify_current_impact", "ask_support_preference"),
        prohibited=("treat_history_as_current_danger",),
        fallback="trauma_support_fallback_v1",
        profile="trauma_support_v1",
    ),
    SituationCode.DOMESTIC_VIOLENCE_SIGNAL_UNSPECIFIED: _definition(
        "domestic_violence_signal_unspecified_v1",
        "Respond without blame to the user's explicit disclosure. For current threats, prioritize privacy and ask whether it is safe to keep chatting or using this device; encourage a trusted person or emergency help if safe. Never advise confronting the person threatening them. If they describe past abuse and say they are safe now, acknowledge that history and ask what support feels useful rather than treating the memory as a current attack. Preserve their control and pacing.",
        "Preserve uncertainty around domestic-violence timing and ask about present safety.",
        goals=("respond_gently_without_assumption", "clarify_immediate_safety", "ask_support_preference"),
        prohibited=("assert_abuse_as_fact", "suggest_confrontation"),
        profile="abuse_uncertain_v1",
        clarification=True,
    ),
    SituationCode.CHILD_ABUSE_CURRENT_DANGER: _definition(
        "child_abuse_current_danger_v1",
        "Treat this as urgent safeguarding outside the validated adult scope. Use simple supportive language and prioritize immediate safety and access to a trusted safe adult or emergency service. Do not promise secrecy, investigate, blame, or make unsupported jurisdiction-specific legal claims.",
        "Apply urgent safeguarding and unsupported-minor-scope constraints without investigating or promising secrecy.",
        tone=("calm", "warm", "direct", "nonjudgmental", "simple"),
        goals=("prioritize_child_safety", "encourage_trusted_safe_adult", "encourage_immediate_local_help"),
        prohibited=("promise_secrecy", "remote_abuse_investigation", "assert_guilt", "invent_jurisdiction_law"),
        required=("worldwide_emergency_fallback", "local_resources_if_eligible", "unsupported_minor_scope_notice"),
        fallback="child_safeguarding_fallback_v1",
        profile="child_safeguarding_v1",
        urgent=True,
    ),
    SituationCode.CHILD_ENDANGERMENT_THIRD_PARTY: _definition(
        "child_endangerment_third_party_v1",
        "Encourage appropriate immediate local safeguarding or emergency help according to approved policy. Do not attempt a remote investigation or assert guilt.",
        "Support appropriate safeguarding for a possibly endangered child without remote investigation or guilt claims.",
        goals=("prioritize_child_safety", "encourage_immediate_local_help", "offer_approved_safeguarding_resources"),
        prohibited=("remote_abuse_investigation", "assert_guilt", "invent_jurisdiction_law"),
        required=("worldwide_emergency_fallback", "local_resources_if_eligible"),
        fallback="child_safeguarding_fallback_v1",
        profile="child_safeguarding_v1",
        urgent=True,
    ),
    SituationCode.CHILDHOOD_ABUSE_HISTORICAL_TRAUMA: _definition(
        "childhood_abuse_historical_trauma_v1",
        "Use trauma-informed support, preserve control and pacing, avoid assuming a child is currently endangered, and ask what would feel helpful now.",
        "Treat historical childhood abuse as trauma support without assuming current child danger.",
        goals=("validate_without_blame", "preserve_user_control", "ask_support_preference"),
        prohibited=("assume_current_child_danger", "request_unnecessary_abuse_details"),
        fallback="trauma_support_fallback_v1",
        profile="trauma_support_v1",
    ),
    SituationCode.CHILD_ABUSE_SIGNAL_UNSPECIFIED: _definition(
        "child_abuse_signal_unspecified_v1",
        "Do not call this active child abuse. Ask one minimal clarifying question about immediate safety and age or scope only when necessary for safe routing.",
        "Preserve age and timing uncertainty and ask only the minimum safeguarding clarification.",
        goals=("respond_gently_without_assumption", "clarify_immediate_safety", "clarify_age_scope_minimally"),
        prohibited=("assert_abuse_as_fact", "remote_abuse_investigation"),
        required=("scope_uncertainty_flag" ,),
        fallback="child_scope_uncertain_fallback_v1",
        profile="child_scope_uncertain_v1",
        clarification=True,
    ),
    SituationCode.POSSIBLE_OVERDOSE_OR_POISONING: _definition(
        "possible_overdose_or_poisoning_v1",
        "Use an immediate urgent medical response. Encourage local emergency or approved poison resources and nearby human help. Do not give dosing or induce-vomiting instructions, and do not wait for RAG or generation before fixed emergency elements appear.",
        "Treat possible overdose or poisoning as urgent; do not give dosing or induce-vomiting instructions.",
        tone=("calm", "direct", "nonjudgmental", "concise"),
        goals=("encourage_immediate_local_help", "encourage_nearby_human_support", "show_approved_poison_resource_if_eligible"),
        prohibited=("provide_dosing_instruction", "instruct_induced_vomiting", "delay_emergency_resources"),
        required=("worldwide_emergency_fallback", "local_resources_if_eligible"),
        fallback="urgent_medical_fallback_v1",
        profile="urgent_medical_v1",
        urgent=True,
    ),
    SituationCode.PANIC_OR_ACUTE_DISTRESS_CURRENT: _definition(
        "panic_or_acute_distress_current_v1",
        "Use short, calm sentences. Acknowledge distress, ask one concise question about severe physical-danger signs or immediate safety, and offer optional grounding. Do not diagnose a panic disorder.",
        "Use short calm support for acute distress without diagnosing; check severe physical danger concisely.",
        tone=("calm", "warm", "concise", "nonjudgmental"),
        goals=("acknowledge_distress", "check_severe_physical_danger", "offer_optional_grounding"),
        prohibited=("diagnose_panic_disorder", "present_grounding_as_emergency_substitute"),
        fallback="acute_distress_fallback_v1",
        profile="acute_distress_v1",
        clarification=True,
    ),
    SituationCode.SUBSTANCE_WITHDRAWAL_OR_ACUTE_SUBSTANCE_CONCERN: _definition(
        "substance_withdrawal_or_acute_concern_v1",
        "Avoid diagnosing severity. Ask only approved concise questions about immediate medical danger and escalate explicit danger. Do not give detox or dosing instructions.",
        "Address possible acute substance concern without diagnosis, detox, or dosing instructions.",
        goals=("check_severe_physical_danger", "encourage_appropriate_professional_help"),
        prohibited=("provide_detox_instruction", "provide_dosing_instruction", "diagnose_withdrawal_severity"),
        fallback="acute_substance_fallback_v1",
        profile="acute_substance_v1",
        clarification=True,
    ),
    SituationCode.ACUTE_CRISIS_SIGNAL_UNSPECIFIED: _definition(
        "acute_crisis_signal_unspecified_v1",
        "Treat the broad signal as monitored uncertainty, not a confirmed emergency. Ask what is happening now and check immediate safety concisely.",
        "Preserve uncertainty around the broad acute-crisis signal and ask what is happening now.",
        goals=("acknowledge_distress", "clarify_immediate_safety"),
        prohibited=("diagnose_from_acute_crisis_label", "assert_emergency_as_fact"),
        fallback="safety_check_fallback_v1",
        profile="crisis_uncertain_v1",
        clarification=True,
    ),
    SituationCode.HARM_OTHERS_ACTIVE_INTENT: _definition(
        "harm_others_active_intent_v1",
        "Be calm and direct. Do not help plan, optimize, or conceal violence. Encourage simple separation from potential targets or weapons and urgent local human or emergency support. Ask one immediate-safety question at a time.",
        "Do not assist violence planning or concealment; prioritize separation and urgent human help.",
        tone=("calm", "direct", "nonjudgmental", "concise"),
        goals=("clarify_immediate_danger_to_others", "encourage_separation_from_targets_or_weapons", "encourage_immediate_local_help"),
        prohibited=("assist_violence_planning", "optimize_or_conceal_violence", "overwhelm_with_questions"),
        required=("worldwide_emergency_fallback", "local_resources_if_eligible"),
        fallback="harm_others_immediate_fallback_v1",
        profile="harm_others_imminent_v1",
        urgent=True,
    ),
    SituationCode.HARM_OTHERS_SIGNAL_UNSPECIFIED: _definition(
        "harm_others_signal_unspecified_v1",
        "Do not assert violent intent. Clarify with one concise question whether anyone is in immediate danger while maintaining calm, nonjudgmental boundaries.",
        "Preserve uncertainty about harm-to-others intent and clarify immediate danger once.",
        goals=("clarify_immediate_danger_to_others", "maintain_nonjudgmental_boundary"),
        prohibited=("assert_violent_intent_as_fact", "assist_violence_planning"),
        fallback="safety_check_fallback_v1",
        profile="harm_others_uncertain_v1",
        clarification=True,
    ),
}


TOPIC_INSTRUCTIONS: Mapping[str, tuple[str, str, tuple[str, ...]]] = {
    "grief": (
        "topic_grief_v1",
        "Use compassionate acknowledgment, avoid timelines or 'move on' language, and ask what support feels useful.",
        ("impose_grief_timeline",),
    ),
    "substance_use": (
        "topic_substance_use_v1",
        "Use non-stigmatizing language and avoid dosing or detox instructions.",
        ("stigmatize_substance_use", "provide_detox_instruction", "provide_dosing_instruction"),
    ),
    "eating_disorder": (
        "topic_eating_disorder_v1",
        "Avoid numbers, competitive detail, weight-loss tips, or reinforcement of harmful behaviors; support appropriate help-seeking.",
        ("provide_eating_disorder_numbers_or_tips", "reinforce_eating_disorder_behavior"),
    ),
    "dissociation": (
        "topic_dissociation_v1",
        "Use calm orienting language and optional grounding without diagnosis; check immediate safety only when context indicates it.",
        ("diagnose_dissociative_condition", "present_grounding_as_treatment"),
    ),
}


BOUNDARY_PROHIBITIONS: Mapping[str, tuple[str, ...]] = {
    "personal_diagnosis_request": ("provide_personal_diagnosis",),
    "third_party_diagnosis_request": ("provide_third_party_diagnosis",),
    "diagnosis_or_clinical_interpretation": ("provide_personal_diagnosis", "provide_personal_clinical_interpretation"),
    "personal_clinical_interpretation": ("provide_personal_clinical_interpretation",),
    "medication_recommendation_or_selection": ("select_or_recommend_medication",),
    "medication_or_personal_treatment": ("select_or_recommend_medication", "provide_personalized_treatment_plan"),
    "personal_use_or_self_treatment": ("provide_personalized_medication_or_treatment_instruction",),
    "dose_or_schedule_instruction": ("provide_personalized_dose_or_schedule",),
    "start_stop_or_switch_medication": ("instruct_start_stop_or_switch_medication",),
    "personalized_treatment_plan": ("provide_personalized_treatment_plan",),
    "harmful_or_unethical_medical_instruction": ("provide_harmful_or_unethical_medical_instruction",),
    "unsafe_medical_practice_or_product": ("provide_harmful_or_unethical_medical_instruction",),
    "professional_role_bypass": ("relax_policy_for_claimed_professional_role",),
    "urgent_medical_safety_request": ("provide_diagnosis_instead_of_urgent_help",),
    "urgent_medical_safety": ("provide_diagnosis_instead_of_urgent_help",),
}


def compose_policy(
    situation_codes: Sequence[SituationCode],
    topic_tags: Iterable[str],
    boundary_route: BoundaryRoute,
    boundary_labels: Iterable[str],
    system_route: SystemRoute,
    *,
    recent_danger_context: bool = False,
    generation_allowed: bool = True,
) -> CompositionResult:
    """Merge policy fragments without concatenating multiple long prompts."""

    definitions = [GUIDANCE[code] for code in situation_codes]
    guidance_ids: list[str] = []
    instructions: list[str] = []
    goals: list[str] = []
    prohibited = list(BASE_PROHIBITIONS)
    required: list[str] = []
    tone: list[str] = []

    for index, definition in enumerate(definitions):
        guidance_ids.append(definition.guidance_id)
        # One full primary instruction plus at most two compact secondary ones.
        if index == 0:
            instructions.append(definition.instruction)
        elif index < 3:
            instructions.append(definition.secondary_instruction)
        _extend_unique(goals, definition.goals)
        _extend_unique(prohibited, definition.prohibited)
        _extend_unique(required, definition.required)
        _extend_unique(tone, definition.tone)

    for tag in sorted(set(topic_tags)):
        topic = TOPIC_INSTRUCTIONS.get(tag)
        if topic is None:
            continue
        guidance_id, instruction, topic_prohibitions = topic
        guidance_ids.append(guidance_id)
        if len(instructions) < 3:
            instructions.append(instruction)
        _extend_unique(prohibited, topic_prohibitions)

    labels = tuple(sorted(set(boundary_labels)))
    for label in labels:
        _extend_unique(prohibited, BOUNDARY_PROHIBITIONS.get(label, ()))

    if boundary_route == BoundaryRoute.SAFE_COMPLETION:
        guidance_ids.append("boundary_safe_completion_v1")
        if len(instructions) < 3:
            instructions.append(
                "State the relevant limitation briefly, offer only general education or help preparing questions for a qualified professional, and do not provide a personalized conclusion or action."
            )
        _extend_unique(goals, ("state_boundary_limit_briefly", "offer_general_information", "suggest_qualified_professional"))
        _extend_unique(required, ("boundary_limit_notice",))
    elif boundary_route == BoundaryRoute.URGENT_MEDICAL_ESCALATION:
        guidance_ids.append("urgent_medical_response_v1")
        if not instructions:
            instructions.append(
                "Give a concise urgent medical response and encourage immediate local help. Exact resource elements are supplied by the application and must appear before generated content."
            )
        _prepend_unique(goals, ("encourage_immediate_local_help", "encourage_nearby_human_support"))
        _extend_unique(required, ("worldwide_emergency_fallback", "local_resources_if_eligible"))
        _extend_unique(prohibited, ("delay_emergency_resources", "provide_personalized_dose_or_schedule"))

    if recent_danger_context:
        guidance_ids.append("recent_danger_followup_v1")
        if len(instructions) < 3:
            instructions.append(
                "Recent session state contains a current or imminent safety signal. Do not treat a later denial or ordinary message as proof that the earlier danger has resolved; check the present situation gently."
            )
        _prepend_unique(goals, ("confirm_current_safety_after_prior_danger",))

    if system_route == SystemRoute.UNSUPPORTED_SCOPE:
        guidance_ids.insert(0, "unsupported_scope_v1")
        instructions.insert(0,
            "Do not generate an ordinary mental-health answer because the current validated scope is English-speaking adults age 18 or older. Use the application-owned scope fallback while preserving any higher-priority urgent safety elements."
        )
        instructions = instructions[:3]
        _prepend_unique(goals, ("state_scope_limit_briefly", "offer_scope_appropriate_next_step"))
        _extend_unique(required, ("scope_limit_notice",))
        generation_allowed = False

    if not guidance_ids:
        guidance_ids.append("normal_support_v1")
        instructions.append(
            "Respond helpfully and warmly within general mental-health information limits. Do not diagnose or provide personalized medication or treatment instructions."
        )
        _extend_unique(tone, ("warm", "respectful"))
        _extend_unique(goals, ("respond_to_user_request",))

    if system_route == SystemRoute.CRISIS_RESPONSE and not definitions:
        guidance_ids.insert(0, "generic_crisis_support_v1")
        _prepend_unique(goals, ("acknowledge_distress", "clarify_immediate_safety", "encourage_nearby_human_support"))

    any_urgent = any(definition.urgent for definition in definitions) or system_route in {
        SystemRoute.CRISIS_RESPONSE,
        SystemRoute.URGENT_MEDICAL_RESPONSE,
    }
    if any_urgent:
        tone = ["calm", "warm", "direct", "nonjudgmental", "concise"]
        _extend_unique(required, ("worldwide_emergency_fallback", "local_resources_if_eligible"))

    # Goal priority follows the already priority-sorted situations.  Boundary
    # urgent goals were prepended.  Three is a deliberate anti-interrogation cap.
    goals = goals[:3]
    primary = definitions[0] if definitions else None
    if system_route == SystemRoute.UNSUPPORTED_SCOPE:
        fallback = "adult_english_scope_fallback_v1"
        profile = "unsupported_scope_v1"
    elif system_route == SystemRoute.URGENT_MEDICAL_RESPONSE and not definitions:
        fallback = "urgent_medical_fallback_v1"
        profile = "urgent_medical_v1"
    elif system_route == SystemRoute.CLINICAL_BOUNDARY_SAFE_COMPLETION and not definitions:
        fallback = "boundary_safe_completion_fallback_v1"
        profile = "boundary_safe_completion_v1"
    elif primary is not None:
        fallback = primary.fallback_template_id
        profile = primary.output_guard_profile
    elif system_route == SystemRoute.MONITORED_SUPPORT:
        fallback = "supportive_unavailable_fallback_v1"
        profile = "supportive_monitored_v1"
    else:
        fallback = "general_unavailable_fallback_v1"
        profile = "supportive_general_v1"

    return CompositionResult(
        generator_guidance_id=guidance_ids[0],
        secondary_guidance_ids=tuple(_dedupe(guidance_ids[1:])),
        guidance_instructions=tuple(instructions),
        tone=tuple(tone or ("warm", "nonjudgmental")),
        response_goals=tuple(goals),
        prohibited_behaviors=tuple(prohibited),
        required_visible_elements=tuple(required),
        needs_clarification=any(definition.needs_clarification for definition in definitions),
        generation_allowed=generation_allowed,
        fixed_content_timing="before_generation" if any_urgent else "with_response",
        fallback_template_id=fallback,
        output_guard_profile=profile,
    )


def guidance_instruction_for(guidance_id: str, *, secondary: bool = False) -> str | None:
    """Resolve only static, allowlisted guidance; never render arbitrary input."""

    for definition in GUIDANCE.values():
        if definition.guidance_id == guidance_id:
            return definition.secondary_instruction if secondary else definition.instruction
    if guidance_id == "generic_crisis_support_v1":
        return "A broad safety signal needs cautious support. Respond to the person's actual words without assuming a crisis category. If they describe current danger, gently ask one immediate-safety question and encourage a trusted nearby person or local emergency help. Continue listening after the fixed contacts; respect explicit negation or historical context."
    if guidance_id == "boundary_safe_completion_v1":
        return (
            "State the relevant limit in one sentence using 'I can't diagnose' or 'I can provide general information'. Then help with safe general education or questions for a qualified professional. A dosing request must not receive a dose, schedule or treatment change. A general educational question should receive a useful explanation, not an unrelated medication refusal."
        )
    if guidance_id == "urgent_medical_response_v1":
        return (
            "If someone is unconscious, unresponsive, has taken an overdose or faces another explicit medical emergency, clearly encourage calling local emergency services now and asking a nearby person for help. Do not wait for answers or suggest grounding instead. Stay calm and direct, and do not diagnose or give remote treatment instructions. The app supplies the exact emergency contacts."
        )
    if guidance_id == "recent_danger_followup_v1":
        return (
            "Use recent conversation to understand the latest update. Acknowledge safety steps and the named support person. Do not silently forget a recent safety concern, but do not repeat questions already answered. If they report feeling able to stay safe with someone present, offer the requested support or optional grounding without asserting that the risk is gone."
        )
    if guidance_id == "normal_support_v1":
        return (
            "Respond helpfully and warmly within general-information limits; do not diagnose or provide personalized medical instructions."
        )
    if guidance_id == "unsupported_scope_v1":
        return (
            "Use the application-owned adult-English scope fallback; do not generate an ordinary answer, while preserving higher-priority urgent safety elements."
        )
    for _, (topic_id, instruction, _) in TOPIC_INSTRUCTIONS.items():
        if topic_id == guidance_id:
            return instruction
    return None


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    _extend_unique(result, values)
    return result


def _extend_unique(target: list[str], values: Iterable[str]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _prepend_unique(target: list[str], values: Iterable[str]) -> None:
    prefix = [value for value in values if value not in target]
    target[:0] = prefix

