from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from mental_health_pipeline.config import PipelineConfig
from mental_health_pipeline.guardrails import ParallelInputGuardrails
from mental_health_pipeline.orchestrator import FullPipeline, PipelineComponents
from mental_health_pipeline.output_guard import PlaceholderOutputGuard
from mental_health_pipeline.resources import ResourceDirectory
from mental_health_pipeline.response_policy import ResponsePolicyBuilder
from mental_health_pipeline.schemas import PipelineRequest, RequestScope
from mental_health_pipeline.session_store import InMemorySafetyStateStore


class Crisis:
    def __init__(self, *, tier="normal", category="no_crisis"):
        self.tier = tier
        self.category = category

    def assess(self, _text):
        categories = [] if self.category == "no_crisis" else [self.category]
        return {
            "tier": self.tier,
            "primary_category": self.category,
            "categories": categories,
            "temporal_status": "current" if self.tier == "crisis" else "unspecified",
            "severity": "urgent" if self.tier == "crisis" else "none",
            "topic_tags": [],
            "classifier": {
                "status": "ok", "predicted_labels": categories, "near_threshold_labels": []
            },
            "rule_trace": {"rule_channel": {"matches": []}},
        }


class Boundary:
    def __init__(self, route="allow", labels=()):
        self.route = route
        self.labels = labels

    def predict(self, _text):
        return {
            "status": "ok",
            "suggested_route": self.route,
            "labels": list(self.labels),
            "needs_review": False,
            "input_coverage": {"complete_input_seen": True},
            "explicit_emergency_screen": {"explicit_signal": False},
        }


class Generator:
    def __init__(self, text="I hear that this has been difficult. What support would help today?"):
        self.text = text
        self.calls = 0

    def readiness(self):
        return {"ready": True}

    def generate(self, _request):
        self.calls += 1
        return SimpleNamespace(text=self.text, citations=())


def pipeline(tmp_path: Path, *, crisis=None, boundary=None, generator=None):
    registry = Path(__file__).resolve().parents[1] / "config" / "emergency_resources.v1.json"
    config = PipelineConfig(
        project_root=tmp_path,
        enable_rag=False,
        enable_generation=True,
        generator_quantization=None,
        output_guard_backend="placeholder",
    )
    components = PipelineComponents(
        input_guardrails=ParallelInputGuardrails(
            crisis or Crisis(), boundary or Boundary(), timeout_seconds=2
        ),
        response_policy=ResponsePolicyBuilder(),
        resources=ResourceDirectory(registry),
        state_store=InMemorySafetyStateStore(),
        output_guard=PlaceholderOutputGuard(),
        generator=generator or Generator(),
    )
    return FullPipeline(config, components)


def request(*, adult=True, country="CA", confirmed=True):
    return PipelineRequest(
        latest_user_message="I have been feeling overwhelmed.",
        session_id="session-1",
        scope=RequestScope(
            age_eligibility_18_plus=adult,
            current_location_country=country,
            location_confirmed=confirmed,
        ),
        request_id="request-1",
    )


def test_normal_turn_is_buffered_and_output_guard_approved(tmp_path: Path) -> None:
    result = pipeline(tmp_path).process(request())
    assert result.route == "normal_chat"
    assert result.used_fallback is False
    assert result.output_guard_status == "approved"
    assert result.resource_payload is None


def test_crisis_resources_are_fixed_before_approved_generated_text(tmp_path: Path) -> None:
    app = pipeline(
        tmp_path,
        crisis=Crisis(tier="crisis", category="suicidal_ideation"),
    )
    result = app.process(request())
    assert result.route == "crisis_response"
    assert result.response.startswith("If you or someone else may be in immediate danger in Canada")
    assert "9-8-8" in result.response
    assert result.resource_payload["country_code"] == "CA"


def test_unconfirmed_location_never_selects_country_resources(tmp_path: Path) -> None:
    app = pipeline(
        tmp_path,
        crisis=Crisis(tier="crisis", category="suicidal_ideation"),
    )
    result = app.process(request(country="CA", confirmed=False))
    assert result.resource_payload["country_code"] is None
    assert "9-8-8" not in result.response
    assert "findahelpline.com" in result.response.casefold()


def test_output_guard_replaces_unsafe_generation_without_leaking_it(tmp_path: Path) -> None:
    unsafe = "You should stop your medication immediately."
    result = pipeline(tmp_path, generator=Generator(unsafe)).process(request())
    assert result.used_fallback is True
    assert unsafe not in result.response
    assert "wasn’t able to produce a response I can trust" in result.response


def test_output_guard_blocks_invented_clinician_identity_without_leaking_it(tmp_path: Path) -> None:
    unsafe = (
        "I am a licensed counselor and I specialize in trauma recovery. "
        "Best wishes, Robin J. Landwehr, DBH, LPC, NCC"
    )
    result = pipeline(tmp_path, generator=Generator(unsafe)).process(request())
    assert result.used_fallback is True
    assert unsafe not in result.response
    assert "licensed counselor" not in result.response
    assert "wasn’t able to produce a response I can trust" in result.response


def test_output_guard_blocks_unsupported_diagnosis_without_leaking_it(tmp_path: Path) -> None:
    unsafe = (
        "I am sorry that you feel depressed. Your depression may be caused by "
        "unresolved trauma."
    )
    result = pipeline(tmp_path, generator=Generator(unsafe)).process(request())
    assert result.used_fallback is True
    assert unsafe not in result.response
    assert "depressed" not in result.response
    assert "wasn’t able to produce a response I can trust" in result.response


def test_unsupported_scope_never_calls_generator(tmp_path: Path) -> None:
    generator = Generator()
    result = pipeline(tmp_path, generator=generator).process(request(adult=False))
    assert result.route == "unsupported_scope"
    assert result.used_fallback is True
    assert generator.calls == 0
    assert "age 18 or older" in result.response
