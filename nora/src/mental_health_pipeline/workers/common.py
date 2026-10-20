"""Strict JSON-line protocol shared by model worker entry points."""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, Mapping


MAX_WORKER_LINE_BYTES = 32_768


def announce_ready(component: str) -> None:
    _send({"schema_version": "worker-ready/1.0", "status": "ready", "component": component})


def serve(component: str, operation: str, handler: Callable[[str], Mapping[str, Any]]) -> int:
    announce_ready(component)
    for line in sys.stdin:
        request_id = "unknown"
        try:
            if len(line.encode("utf-8")) > MAX_WORKER_LINE_BYTES:
                raise ValueError("request_too_large")
            request = json.loads(line)
            if not isinstance(request, Mapping):
                raise ValueError("invalid_request")
            request_id = request.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("invalid_request_id")
            if request.get("schema_version") != "worker-request/1.0":
                raise ValueError("invalid_schema")
            requested_operation = request.get("operation")
            if requested_operation == "shutdown":
                return 0
            if requested_operation != operation:
                raise ValueError("invalid_operation")
            payload = request.get("payload")
            if not isinstance(payload, Mapping) or set(payload) != {"latest_user_message"}:
                raise ValueError("invalid_payload")
            text = payload.get("latest_user_message")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("invalid_message")
            result = handler(text)
            if not isinstance(result, Mapping):
                raise TypeError("invalid_component_result")
            _send({
                "schema_version": "worker-response/1.0",
                "request_id": request_id,
                "status": "ok",
                "result": result,
            })
        except Exception:
            # No exception text: it can contain paths, model internals, or input.
            _send({
                "schema_version": "worker-response/1.0",
                "request_id": request_id,
                "status": "error",
                "error_code": "component_request_failed",
            })
    return 0


def announce_startup_error(component: str) -> None:
    _send({"schema_version": "worker-ready/1.0", "status": "error", "component": component})


def _send(value: Mapping[str, Any]) -> None:
    sys.stdout.write(json.dumps(dict(value), ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()
