"""Privacy-minimal operational telemetry.

This module intentionally has no field for user messages, generated text,
retrieved passages, session identifiers, or exception messages.  Observability
is limited to operational metadata that is useful for capacity and reliability.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Final, Protocol, runtime_checkable


TELEMETRY_SCHEMA_VERSION: Final = "privacy-telemetry/1.0"
_SAFE_TOKEN = re.compile(r"^[A-Za-z0-9._:/-]{1,128}$")


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    event: str
    request_id: str
    outcome: str
    duration_ms: float
    input_bytes: int
    output_bytes: int
    route: str | None = None
    used_fallback: bool = False
    error_code: str | None = None
    schema_version: str = TELEMETRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("event", self.event),
            ("request_id", self.request_id),
            ("outcome", self.outcome),
        ):
            if not isinstance(value, str) or not _SAFE_TOKEN.fullmatch(value):
                raise ValueError(f"{name} must be an opaque safe token")
        for name, value in (("route", self.route), ("error_code", self.error_code)):
            if value is not None and (not isinstance(value, str) or not _SAFE_TOKEN.fullmatch(value)):
                raise ValueError(f"{name} must be an opaque safe token")
        if self.duration_ms < 0:
            raise ValueError("duration_ms must not be negative")
        if self.input_bytes < 0 or self.output_bytes < 0:
            raise ValueError("byte counts must not be negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@runtime_checkable
class TelemetrySink(Protocol):
    def record(self, event: TelemetryEvent) -> None: ...


class PrivacySafeTelemetry:
    """Emit one JSON record without any free-text content."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("mental_health_pipeline.telemetry")

    def record(self, event: TelemetryEvent) -> None:
        if not isinstance(event, TelemetryEvent):
            raise TypeError("only TelemetryEvent instances may be recorded")
        self._logger.info(
            json.dumps(event.to_dict(), ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        )


class NullTelemetry:
    """Explicit no-op sink for tests and installations without log collection."""

    def record(self, event: TelemetryEvent) -> None:
        if not isinstance(event, TelemetryEvent):
            raise TypeError("only TelemetryEvent instances may be recorded")


def byte_length(value: str) -> int:
    """Measure content without retaining or fingerprinting it."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    return len(value.encode("utf-8", errors="strict"))


def error_code_for(exception: BaseException) -> str:
    """Return only the exception class, never its potentially sensitive message."""

    name = type(exception).__name__
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:64]
    return cleaned or "UnhandledError"


__all__ = [
    "NullTelemetry",
    "PrivacySafeTelemetry",
    "TELEMETRY_SCHEMA_VERSION",
    "TelemetryEvent",
    "TelemetrySink",
    "byte_length",
    "error_code_for",
]
