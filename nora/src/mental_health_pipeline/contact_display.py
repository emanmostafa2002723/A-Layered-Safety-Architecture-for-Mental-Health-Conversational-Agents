"""Application contact presentation, separate from a frozen category classifier.

A category/presence prediction is not an assessment of imminent danger. Reuse
the project's existing contextual evidence extractor for contact presentation;
never modify a classifier result or treat absence of a rule as evidence of safety.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
import importlib
from pathlib import Path
import sys
import threading
from types import ModuleType


MARKERS = frozenset({"worldwide_emergency_fallback", "local_resources_if_eligible"})
_IMPORT_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def _extractor(component_root: Path):
    directory = component_root.resolve() / "src/input_guardrail"
    if not (directory / "rules.py").is_file() or not (directory / "schemas.py").is_file():
        return None
    # Give the two existing, dependency-free files a private package name.
    # Importing a legacy top-level `src` package would collide with classifiers.
    name = "_mhp_contact_evidence_" + hashlib.sha256(str(directory).encode()).hexdigest()[:12]
    with _IMPORT_LOCK:
        if name not in sys.modules:
            package = ModuleType(name)
            package.__path__ = [str(directory)]
            sys.modules[name] = package
        return importlib.import_module(name + ".rules").extract_rules


def explicit_evidence(text: str, component_root: Path | None):
    if component_root is None:
        return ()
    extract = _extractor(component_root)
    return tuple(extract(text.casefold().replace("’", "'")).active_matches) if extract else ()


def apply_contact_display_policy(packet, user_text: str, component_root: Path | None, *, history=()):
    """Gate the long emergency block; retain routing, labels and raw decisions."""
    matches = explicit_evidence(user_text, component_root)
    urgent_matches = [m for m in matches if m.severity.value == "urgent"
                      and m.temporal.value not in {"past", "recent"}]
    # Preserve trusted urgency metadata from a contextual upstream contract.
    # ST01c explicitly supplies severity=none/timing=unspecified, so a binary
    # category override cannot independently satisfy this condition.
    trusted_immediate = (packet.severity.value == "urgent"
                         and packet.temporal_status.value in {"current", "imminent", "mixed"})
    medical = packet.system_route.value == "urgent_medical_response"
    emergency = bool(urgent_matches) or trusted_immediate or medical
    suicide_emergency = emergency and (
        any(m.category == "suicidal_ideation" for m in urgent_matches)
        or (trusted_immediate and "suicidal_ideation" in packet.crisis_labels)
    )
    required = tuple(x for x in packet.required_visible_elements if x not in MARKERS)
    if emergency:
        required += tuple(sorted(MARKERS))
    mode = "suicide_emergency" if suicide_emergency else "other_emergency" if emergency else "no_emergency_block"
    flags = (*packet.policy_flags, "contact_display:" + mode)
    fallback = packet.fallback_template_id
    supported_categories = {m.category for m in matches if m.category}
    for old in history[-8:]:
        if old.get("role") == "user":
            supported_categories.update(m.category for m in explicit_evidence(old["content"], component_root) if m.category)
    # The child's-scope question is conditional guidance, not a universal
    # requirement when an uncertain classifier mislabeled an adult's message.
    # Preserve the model's label/route for audit; never invent a child to make
    # an output checker pass. Actual safeguarding signals retain their rules.
    if ('scope_uncertainty_flag' in required
            and 'child_abuse_or_endangerment' not in supported_categories
            and packet.generator_guidance_id != 'final_boundary_child_v1'):
        required=tuple(x for x in required if x!='scope_uncertainty_flag')
        flags+=('conditional_child_scope_not_grounded_in_user_context',)
    # Conditional category guidance must not become an unconditional, scripted
    # assertion of suicidality when generation is rejected for an unrelated topic.
    if (fallback == "safety_check_fallback_v1" and not emergency
            and "suicidal_ideation" not in supported_categories
            and packet.temporal_status.value == "unspecified"):
        if "self_directed_harm" in supported_categories:
            fallback = "self_harm_support_fallback_v1"
        elif {"domestic_violence", "sexual_violence"}.intersection(supported_categories):
            fallback = "trauma_support_fallback_v1"
        else:
            fallback = "supportive_unavailable_fallback_v1"
        flags += ("unconfirmed_category_fallback_generalized",)
    return replace(packet, required_visible_elements=required,
                   fixed_content_timing="before_generation" if emergency else "with_response",
                   fallback_template_id=fallback, policy_flags=flags)
