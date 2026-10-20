"""Opt-in, in-memory diagnostics for the full pipeline's private test interface.

The normal HTTP API is unchanged. This console observes the same orchestration
and never returns an unapproved generator draft or bypasses a safety decision.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
import threading
import time
from typing import Any, Mapping
import uuid

from .orchestrator import FullPipeline
from .schemas import PipelineRequest, PipelineResponse, RequestScope


def json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return type(value).__name__


class TurnRecorder:
    """One active turn; classifier threads share this recorder under a lock."""

    def __init__(self):
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []
        self.active = False
        self.token = None

    def begin(self):
        with self._lock:
            self.events = []
            self.active = True
            self.token = object()

    def capture(self):
        with self._lock:
            return self.token if self.active else None

    def finish(self):
        with self._lock:
            self.active = False
            return list(self.events)

    def append(self, event, token):
        with self._lock:
            if self.active and token is self.token:
                self.events.append(event)


def _summary(component, result, args):
    if component == "final_gate":
        return {"action": result.get("action"), "reason": result.get("reason"),
                "route": result.get("route"), "text_released": result.get("action") in {"DISPLAY","DISPLAY_CHECKED_REPLACEMENT"},
                "replacement_checked":result.get('replacement_checked',False),
                "original_reason_codes":result.get('original_reason_codes',[]),
                "original_medical_check":result.get('original_medical_check'),
                "original_blocking_stage":result.get('original_blocking_stage'),
                "guard_version":result.get('guard_version'),"reason_codes":result.get('reason_codes',[]),
                "checks":result.get('diagnostics',{})}
    if component == "rag":
        return {
            "retrieval_query": result.retrieval_query,
            "skipped_reason": result.skipped_reason,
            "mode": result.mode.value,
            "diagnostics": json_value(result.diagnostics),
            "documents": [{
                "citation": json_value(doc.citation), "score": doc.score,
                "text_preview": doc.text[:2400], "full_text_characters": len(doc.text),
            } for doc in result.documents],
        }
    if component == "generator":
        request = args[0]
        retrieval = request.retrieval
        effective=getattr(result,'effective_retrieval',None) if getattr(result,'evidence_snapshot_available',False) else None
        return {
            "evidence_supplied_to_generator": bool(retrieval and retrieval.documents),
            "context_used_in_final_prompt": getattr(result, "context_used", None),
            "prompt_tokens": getattr(result, "prompt_tokens", None),
            "generated_tokens": getattr(result, "generated_tokens", None),
            "complete": getattr(result, "complete", None),
            "cited_source_ids": [c.source_id for c in result.citations],
            "evidence_snapshot_available":getattr(result,'evidence_snapshot_available',False),
            "effective_evidence_sources":[json_value(doc.citation) for doc in effective.documents] if effective else [],
            "policy_instructions_sent": request.response_policy,
            "turn_specific_policy_present": bool(request.response_policy.strip()),
            "turn_specific_policy_characters": len(request.response_policy),
            "history_messages_supplied": len(request.history),
            "personal_memory_notes_supplied": len(request.personal_memory),
            "note": "Context inclusion and citation markers do not prove causal influence. Draft text is withheld.",
        }
    if component == "output_guard":
        # Never serialize any candidate/response text from a guard implementation.
        return {key: json_value(getattr(result, key, None)) for key in (
            "action", "reason_codes", "guard_version", "evaluated_complete_response",
        )}
    return json_value(result)


class ObservedComponent:
    def __init__(self, target, component, methods, recorder):
        self._target = target
        self._component = component
        self._methods = frozenset(methods)
        self._recorder = recorder

    def __getattr__(self, name):
        original = getattr(self._target, name)
        if name not in self._methods or not callable(original):
            return original

        def invoke(*args, **kwargs):
            token = self._recorder.capture()
            started = time.perf_counter()
            status, error_type, details = "ok", None, None
            try:
                result = original(*args, **kwargs)
                try:
                    details = _summary(self._component, result, args)
                except Exception:
                    # Diagnostics must never change a successful pipeline result.
                    details = {"diagnostic_summary_unavailable": True}
                return result
            except Exception as error:
                status, error_type = "error", type(error).__name__
                raise
            finally:
                self._recorder.append({
                    "component": self._component, "method": name,
                    "seconds": time.perf_counter() - started,
                    "status": status, "error_type": error_type, "details": details,
                }, token)
        return invoke


@dataclass(frozen=True)
class ConsoleResult:
    response: PipelineResponse
    diagnostics: dict[str, Any]


class TestingConsole:
    """Serialize all UI turns around one persistent, instrumented pipeline."""

    __test__ = False

    def __init__(self, pipeline: FullPipeline):
        self._lock = threading.RLock()
        self.recorder = TurnRecorder()
        c = pipeline.components
        observe = lambda target, name, methods: ObservedComponent(target, name, methods, self.recorder)
        guards = c.input_guardrails
        # Keep the original parallel executor and its timeout/fail-closed logic.
        guards.crisis_guardrail = observe(guards.crisis_guardrail, "crisis", ("assess",))
        guards.boundary_guardrail = observe(guards.boundary_guardrail, "boundary", ("predict",))
        pipeline.components = replace(
            c,
            input_guardrails=observe(guards, "input_guardrails_parallel", ("assess",)),
            response_policy=observe(c.response_policy, "response_policy", ("build",)),
            resources=observe(c.resources, "resources", ("resolve",)),
            rag=observe(c.rag, "rag", ("retrieve",)) if c.rag is not None else None,
            generator=observe(c.generator, "generator", ("generate",)) if c.generator is not None else None,
            output_guard=observe(c.output_guard, "output_guard", ("inspect", "inspect_with_context")),
            final_gate=observe(c.final_gate, "final_gate", ("deliver",)) if c.final_gate is not None else None,
        )
        pipeline._prepare_display_policy = observe(
            pipeline, "contact_display_policy", ("_prepare_display_policy",)
        )._prepare_display_policy
        pipeline._inspect_conversation_quality = observe(
            pipeline, "conversation_quality", ("_inspect_conversation_quality",)
        )._inspect_conversation_quality
        self.pipeline = pipeline
        self.turns_completed = 0

    def process(self, request: PipelineRequest) -> ConsoleResult:
        with self._lock:
            self.recorder.begin()
            started = time.perf_counter()
            try:
                response = self.pipeline.process(request)
            finally:
                elapsed = time.perf_counter() - started
                events = self.recorder.finish()
            self.turns_completed += 1
            details = {name: [e["details"] for e in events if e["component"] == name]
                       for name in ("crisis", "boundary", "response_policy", "contact_display_policy",
                                    "rag", "generator", "output_guard", "conversation_quality", "final_gate")}
            rows = [[e["component"], e["method"], round(e["seconds"], 4),
                     e["status"] if e["error_type"] is None else e["error_type"]] for e in events]
            rag = details["rag"][-1] if details["rag"] else {}
            diagnostics = rag.get("diagnostics", {}) if isinstance(rag, dict) else {}
            backend=diagnostics.get('router_backend','qwen_v4')
            router_label=('lightweight_router (inside rag)' if backend=='legacy' else
                          'qwen_router (inside rag)' if str(backend).startswith('qwen') else
                          str(backend)+'_router (inside rag)')
            for key, label in (("router_latency_ms", router_label),
                               ("retrieval_latency_ms", "retrieval_and_reranker (inside rag)")):
                milliseconds = diagnostics.get(key)
                if isinstance(milliseconds, (int, float)):
                    rows.append([label, "retrieve", round(milliseconds / 1000, 4), "ok"])
            rows.append(["full_pipeline", "process", round(elapsed, 4), "complete"])
            return ConsoleResult(response, {
                "request_id": response.request_id,
                "total_seconds": round(elapsed, 4), "console_turn": self.turns_completed,
                "first_console_turn": self.turns_completed == 1,
                "output_guard_status": response.output_guard_status,
                "generator_adapter": getattr(self.pipeline.config.mistral_adapter, 'name', None),
                "generator_prompt_profile": self.pipeline.config.generator_prompt_profile,
                "execution_experiments": self.pipeline.execution_diagnostics,
                "incremental_output": getattr(self.pipeline.components.generator,
                    'last_incremental_diagnostics', {'mode':'unavailable'}),
                "timings": rows, "components": details, "events": events,
                "conversation_signal_indicator":self.pipeline.components.state_store.get(
                    request.session_id).conversation_indicator(),
                "timing_note": "Server wall time only; excludes browser queue/network. Classifiers run in parallel; nested rows overlap. First calls may include model loading. Do not sum rows.",
            })

    def new_session(self, previous_session: str | None = None) -> dict[str, Any]:
        with self._lock:
            if previous_session:
                self.pipeline.conversation_store.delete(previous_session)
                self.pipeline.components.state_store.delete(previous_session)
        return {"id": "gradio-" + uuid.uuid4().hex, "turn": 0}

    def close(self):
        with self._lock:
            self.pipeline.close()


def make_request(message: str, state: Mapping[str, Any], adult: bool,
                 country: str, confirmed: bool) -> PipelineRequest:
    if adult is not True:
        raise ValueError("Confirm the adult test scope before sending.")
    request = PipelineRequest(
        latest_user_message=message, session_id=str(state["id"]),
        turn_number=int(state["turn"]) + 1,
        request_id="gradio-" + uuid.uuid4().hex,
        scope=RequestScope(age_eligibility_18_plus=True, preferred_language="en",
                           current_location_country=country.strip().upper() or None,
                           location_confirmed=bool(confirmed)),
    )
    request.validate()
    return request
