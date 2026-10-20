from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from mental_health_pipeline.guardrails import bridge_static_crisis
from mental_health_pipeline.schemas import PipelineRequest, RequestScope
from test_orchestrator import Generator, pipeline


ROOT = Path(__file__).parents[1]


class FalseSuicideSignal:
    """Reproduce the display failure without assuming the live classifier's cause."""
    def assess(self, text):
        return bridge_static_crisis({"schema_version": "crisis-static-v2",
            "action": "crisis_specific_route", "detected_categories": ["suicidal_ideation"],
            "truncated": False})


def app(tmp_path, generator=None):
    result = pipeline(tmp_path, crisis=FalseSuicideSignal(), generator=generator)
    result.config = replace(result.config, guardrail_repo=ROOT / "components/context_rules")
    return result


def message(text, turn=1):
    return PipelineRequest(text, "conversation-repair", RequestScope(
        age_eligibility_18_plus=True, current_location_country="CA", location_confirmed=True), turn)


@pytest.mark.parametrize("text", [
    "hi", "I feel nervous", "i have a test towmorro", "i have an exam tomorrow",
    "i have an interview", "fuck you", "i feel sad", "because i faild my exam",
    "I failed and my father burend me in my arm", "i cut myself last night",
    "I had suicidal thoughts years ago. I feel safe now.",
    "I am not going to kill myself now.",
    "What does suicide prevention mean?",
])
def test_broad_static_category_does_not_authorize_emergency_banner(tmp_path, text):
    result = app(tmp_path).process(message(text))
    assert result.route == "crisis_response"  # No classifier downgrade or threshold change.
    assert result.resource_payload is None
    assert "9-1-1" not in result.response and "9-8-8" not in result.response


@pytest.mark.parametrize("text", [
    "I am going to kill myself now.",
    "I have a plan to end my life tonight.",
])
def test_current_explicit_suicide_intent_keeps_fixed_emergency_contacts(tmp_path, text):
    result = app(tmp_path).process(message(text))
    assert result.resource_payload is not None
    assert "9-8-8" in result.response


def test_recent_self_harm_does_not_get_suicide_specific_script(tmp_path):
    result = app(tmp_path, Generator("You should stop your medication immediately.")).process(message("i cut myself last night"))
    assert result.used_fallback
    assert "medical attention" in result.response
    assert "acting on these thoughts" not in result.response
    assert result.resource_payload is None


def test_others_inflicting_injury_does_not_get_assumed_suicide_fallback(tmp_path):
    result = app(tmp_path, Generator("You should stop your medication immediately.")).process(
        message("I failed and my father burend me in my arm"))
    assert result.used_fallback
    assert "acting on these thoughts" not in result.response
    assert result.resource_payload is None


class ReplySequence(Generator):
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return SimpleNamespace(text=next(self.replies), citations=())


def test_copied_reply_is_rewritten_before_display_and_latest_input_is_preserved(tmp_path):
    stale = "I'm sorry to hear that. What is making you nervous?"
    fresh = "An exam tomorrow can feel stressful. Which subject would you like to make a short revision plan for?"
    generator = ReplySequence([stale, stale, fresh])
    instance = pipeline(tmp_path, generator=generator)
    assert instance.process(message("I feel nervous")).response == stale
    result = instance.process(message("i have an exam tomorrow", 2))
    assert result.response == fresh
    assert not result.used_fallback
    assert len(generator.requests) == 3
    assert generator.requests[-1].user_text == "i have an exam tomorrow"
    assert "Write a fresh answer" in generator.requests[-1].response_policy
    assert generator.requests[-1].history[-2]["content"] == "I feel nervous"


def test_repeated_apology_is_rewritten_even_when_question_changes(tmp_path):
    generator = ReplySequence([
        "I'm sorry to hear that. What is making you feel sad?",
        "I'm sorry to hear that. What happened?",
        "Failing an exam can feel disappointing. Would reviewing one difficult topic together help?",
    ])
    instance = pipeline(tmp_path, generator=generator)
    instance.process(message("i feel sad"))
    result = instance.process(message("because i faild my exam", 2))
    assert not result.response.startswith("I'm sorry")
    assert not result.used_fallback


def test_persistent_copying_is_not_silently_approved(tmp_path):
    stale = "I'm sorry to hear that. What is making you nervous?"
    generator = ReplySequence([stale, stale, stale])
    instance = pipeline(tmp_path, generator=generator)
    instance.process(message("I feel nervous"))
    result = instance.process(message("i have an interview", 2))
    assert result.used_fallback
    assert result.response != stale
    assert "repeated_previous_response" in result.output_guard_status


def test_invented_suicidal_thoughts_receive_one_contextual_rewrite(tmp_path):
    generator = ReplySequence([
        "Are you at risk of acting on these thoughts right now? I can keep talking with you while you reach out.",
        "That sounds painful, and you did not deserve to be hurt. Are you somewhere safe away from your father, and can you get medical help for your arm?",
    ])
    result = app(tmp_path, generator).process(message("My father burned my arm."))
    assert not result.used_fallback
    assert "acting on these thoughts" not in result.response
    assert "arm" in result.response


# The old ZIP-installer test is preserved verbatim under
# docs/history/previous_release_tests/test_conversation_repair.py.previous.
# This candidate is assembled directly and does not ship those old installers.
