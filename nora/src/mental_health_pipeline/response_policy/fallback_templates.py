"""Versioned non-generative fallbacks for generator/output-guard failure.

These are conservative engineering templates.  They have not received the
clinical, legal, accessibility, localization, or human-factors approval needed
for unrestricted public deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .schemas import ResponsePolicyPacket


FALLBACK_TEMPLATE_VERSION = "response-fallbacks-2026-10-14"


@dataclass(frozen=True, slots=True)
class FallbackTemplate:
    template_id: str
    text: str
    required_visible_elements: tuple[str, ...] = ()
    template_version: str = FALLBACK_TEMPLATE_VERSION
    public_deployment_approved: bool = False


@dataclass(frozen=True, slots=True)
class FallbackResponse:
    text: str
    template_id: str
    template_version: str
    required_visible_elements: tuple[str, ...]
    public_deployment_approved: bool = False


TEMPLATES: Mapping[str, FallbackTemplate] = {
    "immediate_safety_fallback_v1": FallbackTemplate(
        "immediate_safety_fallback_v1",
        "I’m glad you told me. If you may act now or are in immediate danger, contact your local emergency service or go to the nearest emergency department now. If you can, ask a trusted person to stay with you while you get help. Are you in immediate danger right now?",
        ("worldwide_emergency_fallback", "local_resources_if_eligible"),
    ),
    "safety_check_fallback_v1": FallbackTemplate(
        "safety_check_fallback_v1",
        "Thank you for telling me. We can take this one step at a time. Are you at risk of acting on these thoughts right now, or have you already hurt yourself? If you may act now, please seek immediate local help. If you can, put some distance between yourself and anything you could use to hurt yourself, and ask someone you trust to stay with you. I can keep talking with you while you reach out.",
    ),
    "self_harm_current_fallback_v1": FallbackTemplate(
        "self_harm_current_fallback_v1",
        "I’m sorry you’re going through this. If you are injured or still hurting yourself, please contact local emergency help now and ask someone nearby to stay with you. Do you need urgent medical attention right now?",
        ("worldwide_emergency_fallback", "local_resources_if_eligible"),
    ),
    "self_harm_support_fallback_v1": FallbackTemplate(
        "self_harm_support_fallback_v1",
        "Thank you for telling me. You deserve support without judgment. Are you physically safe right now, and do any injuries need medical attention?",
    ),
    "abuse_current_danger_fallback_v1": FallbackTemplate(
        "abuse_current_danger_fallback_v1",
        "What is happening is not your fault. If you are in immediate danger and it is safe to do so, move toward a safer place and contact local emergency help or a trusted person. Is it safe for you to keep chatting?",
        ("worldwide_emergency_fallback", "local_resources_if_eligible"),
    ),
    "domestic_violence_current_fallback_v1": FallbackTemplate(
        "domestic_violence_current_fallback_v1",
        "This is not your fault. Your safety and privacy come first. If it is safe to do so, move toward a safer place and contact local emergency help or someone you trust. Is it safe for you to keep using this device?",
        ("worldwide_emergency_fallback", "local_resources_if_eligible", "privacy_safety_reminder"),
    ),
    "trauma_support_fallback_v1": FallbackTemplate(
        "trauma_support_fallback_v1",
        "I’m sorry this happened to you. You can choose how much to share and what kind of support feels useful right now. We can take this one step at a time.",
    ),
    "child_safeguarding_fallback_v1": FallbackTemplate(
        "child_safeguarding_fallback_v1",
        "You deserve to be safe, and this is not your fault. If there is immediate danger, contact local emergency help now. If you can do so safely, tell a trusted safe adult who can be with you and help.",
        ("worldwide_emergency_fallback", "local_resources_if_eligible", "unsupported_minor_scope_notice"),
    ),
    "child_scope_uncertain_fallback_v1": FallbackTemplate(
        "child_scope_uncertain_fallback_v1",
        "I want to respond safely without making assumptions. Is a child in immediate danger right now, and are you asking about yourself or someone else?",
        ("scope_uncertainty_flag",),
    ),
    "urgent_medical_fallback_v1": FallbackTemplate(
        "urgent_medical_fallback_v1",
        "This may need urgent medical attention. Contact your local emergency service or go to the nearest emergency department now. If possible, ask someone nearby to stay with you. Do not wait here for a diagnosis or dosing instruction.",
        ("worldwide_emergency_fallback", "local_resources_if_eligible"),
    ),
    "acute_distress_fallback_v1": FallbackTemplate(
        "acute_distress_fallback_v1",
        "I’m here with you. If you have severe breathing difficulty, chest pain, loss of consciousness, or another immediate physical danger, contact local emergency help now. Otherwise, if it feels comfortable, place your feet on the floor and notice one thing you can see while we take this slowly.",
    ),
    "acute_substance_fallback_v1": FallbackTemplate(
        "acute_substance_fallback_v1",
        "If there is trouble breathing, loss of consciousness, a seizure, severe confusion, or another immediate danger, contact local emergency help now and ask someone nearby to stay. I can’t safely provide dosing or detox instructions.",
    ),
    "harm_others_immediate_fallback_v1": FallbackTemplate(
        "harm_others_immediate_fallback_v1",
        "I can’t help plan or carry out harm. Move away from anyone who could be hurt and from any weapon if you can do so safely, and contact local emergency help or a trusted person now. Is anyone in immediate danger?",
        ("worldwide_emergency_fallback", "local_resources_if_eligible"),
    ),
    "boundary_safe_completion_fallback_v1": FallbackTemplate(
        "boundary_safe_completion_fallback_v1",
        "I can offer general information, but I can’t determine a diagnosis or recommend a personalized medication, dose, schedule, or treatment change. A qualified clinician or pharmacist can help with your specific situation.",
        ("boundary_limit_notice",),
    ),
    "adult_english_scope_fallback_v1": FallbackTemplate(
        "adult_english_scope_fallback_v1",
        "This version is currently designed for English-speaking adults age 18 or older, so I can’t provide the ordinary chat response here. If someone may be in immediate danger, contact local emergency help now; otherwise, please use an age- and language-appropriate qualified service.",
        ("scope_limit_notice",),
    ),
    "supportive_unavailable_fallback_v1": FallbackTemplate(
        "supportive_unavailable_fallback_v1",
        "I’m here to support you, but I wasn’t able to produce a response I can trust. Please try again or rephrase what you’d like help with. If you need human support now, consider contacting someone you trust or a qualified mental-health professional.",
    ),
    "general_unavailable_fallback_v1": FallbackTemplate(
        "general_unavailable_fallback_v1",
        "I wasn’t able to produce a response I can trust. Please try again or rephrase what you’d like help with.",
    ),
}


def get_fallback_template(template_id: str) -> FallbackTemplate:
    try:
        return TEMPLATES[template_id]
    except KeyError as error:
        raise ValueError(f"Unknown fallback template: {template_id!r}") from error


def render_fallback(packet: ResponsePolicyPacket) -> FallbackResponse:
    """Render the packet-selected fallback plus all fixed element contracts."""

    template = get_fallback_template(packet.fallback_template_id)
    elements = tuple(dict.fromkeys(
        (*packet.required_visible_elements, *template.required_visible_elements)
    ))
    return FallbackResponse(
        text=template.text,
        template_id=template.template_id,
        template_version=template.template_version,
        required_visible_elements=elements,
        public_deployment_approved=template.public_deployment_approved,
    )

