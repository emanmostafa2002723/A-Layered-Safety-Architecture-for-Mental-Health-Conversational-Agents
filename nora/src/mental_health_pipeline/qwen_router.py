"""Pinned, request-local Qwen V4 retrieval routing over loopback Ollama.

Never import the evaluation scripts: they write files and mutate a global
prompt. This adapter uses their exact frozen prompt/schema, without retries,
downloads, raw-text logging, or authority over the safety route.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
import urllib.request


PROMPT_SHA256 = "5b532906f8fef5c64d627a99002f261ab0797b77d4a6f2120dd15b4f17a0284b"
MODEL_DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
REASONS = frozenset({"mental_health_knowledge", "emotional_support", "social_message",
                     "live_information", "unsupported_scope", "general_conversation_or_creation"})
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "retrieval_decision": {"type": "string", "enum": ["RETRIEVE", "NO_RETRIEVE"]},
        "canonical_query": {"type": "string"},
        "reason_code": {"type": "string", "enum": sorted(REASONS)},
    },
    "required": ["retrieval_decision", "canonical_query", "reason_code"],
    "additionalProperties": False,
}


class QwenRouterError(RuntimeError):
    """Only fixed codes may reach health responses or operational metadata."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class QwenRoute:
    decision: str
    canonical_query: str
    reason_code: str
    latency_ms: float = 0.0
    application_override: str | None = None
    raw_decision: str | None = None

    @property
    def requires_retrieval(self) -> bool:
        return self.decision == "RETRIEVE"


def validate_result(value: object) -> QwenRoute:
    if not isinstance(value, dict) or set(value) != set(OUTPUT_SCHEMA["required"]):
        raise QwenRouterError("qwen_invalid_output")
    decision, query, reason = (value[k] for k in OUTPUT_SCHEMA["required"])
    if not all(isinstance(v, str) for v in (decision, query, reason)):
        raise QwenRouterError("qwen_invalid_output")
    query = " ".join(query.split())
    if (decision not in {"RETRIEVE", "NO_RETRIEVE"} or reason not in REASONS
            or len(query) > 4096
            or (decision == "RETRIEVE" and (len(query) < 5 or reason != "mental_health_knowledge"))
            or (decision == "NO_RETRIEVE" and (query or reason == "mental_health_knowledge"))):
        raise QwenRouterError("qwen_invalid_output")
    return QwenRoute(decision, query, reason)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise QwenRouterError("qwen_redirect_refused")


class QwenJointRouter:
    def __init__(self, prompt_path: Path, *, api_url="http://127.0.0.1:11434/api/chat",
                 model="qwen3:4b", timeout_seconds=30.0, num_gpu=0, keep_alive="5m", context_tokens=4096):
        parsed = urlsplit(api_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path != "/api/chat"):
            raise ValueError("Qwen endpoint must be loopback HTTP /api/chat without credentials")
        if not 0 < timeout_seconds <= 180 or num_gpu < -1:
            raise ValueError("Invalid Qwen timeout or GPU-layer setting")
        if type(context_tokens) is not int or not 4096<=context_tokens<=16384:
            raise ValueError('Invalid Qwen context budget')
        self.prompt_path = Path(prompt_path)
        self.api_url, self.model = api_url, model
        self.tags_url = parsed._replace(path="/api/tags").geturl()
        self.timeout_seconds = timeout_seconds
        self.num_gpu, self.keep_alive = num_gpu, keep_alive
        self.context_tokens=context_tokens
        self._slot = threading.BoundedSemaphore(1)
        # Environment HTTP proxies must never forward private messages off-host.
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _prompt(self) -> str:
        try:
            raw = self.prompt_path.read_bytes()
        except OSError:
            raise QwenRouterError("qwen_prompt_missing") from None
        if hashlib.sha256(raw).hexdigest() != PROMPT_SHA256:
            raise QwenRouterError("qwen_prompt_hash_mismatch")
        return raw.decode("utf-8")

    def _request_json(self, url: str, *, payload=None, timeout: float):
        request = urllib.request.Request(
            url, data=None if payload is None else json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = response.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise QwenRouterError("qwen_response_too_large")
            result = json.loads(raw)
            if not isinstance(result, dict) or "error" in result:
                raise QwenRouterError("qwen_invalid_response")
            return result
        except QwenRouterError:
            raise
        except Exception:
            # Network/provider errors can contain input text: never expose them.
            raise QwenRouterError("qwen_service_unavailable") from None

    def _identity(self, timeout: float) -> None:
        data = self._request_json(self.tags_url, timeout=timeout)
        models = data.get("models")
        if not isinstance(models, list):
            raise QwenRouterError("qwen_invalid_response")
        found = next((m for m in models if isinstance(m, dict)
                      and self.model in (m.get("name"), m.get("model"))), None)
        if found is None:
            raise QwenRouterError("qwen_model_missing")
        digest = found.get("digest")
        if not isinstance(digest, str) or digest.removeprefix("sha256:") != MODEL_DIGEST:
            raise QwenRouterError("qwen_model_digest_mismatch")

    def readiness_code(self) -> str | None:
        """Check local prompt + installed full digest, without loading Qwen."""
        try:
            self._prompt()
            self._identity(min(self.timeout_seconds, 3.0))
            return None
        except QwenRouterError as error:
            return error.code

    def route(self, original_query: str) -> QwenRoute:
        if not isinstance(original_query, str) or not original_query.strip():
            raise ValueError("Qwen requires a non-empty message")
        if len(original_query.encode("utf-8")) > 16384:
            raise QwenRouterError("qwen_input_limit")
        started = time.monotonic()
        deadline = started + self.timeout_seconds
        if not self._slot.acquire(timeout=self.timeout_seconds):
            raise QwenRouterError("qwen_busy")
        try:
            prompt = self._prompt()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QwenRouterError("qwen_timeout")
            self._identity(min(remaining, 3.0))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise QwenRouterError("qwen_timeout")
            result = self._request_json(self.api_url, timeout=remaining, payload={
                "model": self.model, "stream": False, "think": False,
                "format": OUTPUT_SCHEMA, "keep_alive": self.keep_alive,
                "messages": [{"role": "system", "content": prompt},
                             {"role": "user", "content": original_query}],
                "options": {"temperature": 0, "num_predict": 160, "num_gpu": self.num_gpu,
                            "num_ctx": self.context_tokens},
            })
            if time.monotonic() > deadline:
                raise QwenRouterError("qwen_timeout")
            if result.get("done") is not True or result.get("done_reason") == "length":
                raise QwenRouterError("qwen_incomplete_response")
            try:
                message = result["message"]
                if message.get("role") != "assistant":
                    raise QwenRouterError("qwen_invalid_output")
                value = json.loads(message["content"])
            except (KeyError, TypeError, AttributeError, ValueError):
                raise QwenRouterError("qwen_invalid_output") from None
            route = validate_result(value)
            return QwenRoute(route.decision, route.canonical_query, route.reason_code,
                             (time.monotonic() - started) * 1000)
        finally:
            self._slot.release()
