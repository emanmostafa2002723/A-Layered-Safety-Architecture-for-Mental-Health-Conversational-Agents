"""Audio-transcription boundary; no speech model is currently supplied."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


MAX_AUDIO_BYTES = 25 * 1024 * 1024
MAX_TRANSCRIPT_BYTES = 16_384


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    language: str
    provider_version: str
    complete: bool = True
    schema_version: str = "transcription-result/1.0"

    def __post_init__(self) -> None:
        if self.schema_version != "transcription-result/1.0":
            raise ValueError("unsupported transcription schema")
        if not self.complete:
            raise ValueError("partial transcripts cannot enter the safety pipeline")
        if not self.text.strip():
            raise ValueError("transcript must not be blank")
        if len(self.text.encode("utf-8")) > MAX_TRANSCRIPT_BYTES:
            raise ValueError("transcript exceeds the pipeline message limit")
        if self.language.casefold() != "en":
            raise ValueError("this release accepts English transcripts only")


class TranscriptionProvider(Protocol):
    production_ready: bool

    def transcribe(self, audio: bytes, *, media_type: str) -> TranscriptionResult: ...

    def readiness(self) -> dict[str, object]: ...


class UnavailableTranscriptionProvider:
    """Explicit placeholder used until a tested ASR component is supplied."""

    production_ready = False
    version = "transcription-unavailable/1.0"

    def transcribe(self, audio: bytes, *, media_type: str) -> TranscriptionResult:
        del audio, media_type
        raise RuntimeError("transcription_backend_unavailable")

    def readiness(self) -> dict[str, object]:
        return {
            "ready": False,
            "component": "transcription",
            "version": self.version,
            "issues": ["no_transcription_model_or_service_has_been_supplied"],
        }


def validate_audio_input(audio: bytes, *, media_type: str) -> None:
    if not isinstance(audio, bytes) or not audio:
        raise ValueError("audio must be non-empty bytes")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("audio exceeds the upload limit")
    if media_type not in {"audio/wav", "audio/mpeg", "audio/mp4", "audio/webm", "audio/ogg"}:
        raise ValueError("unsupported audio media type")


__all__ = [
    "MAX_AUDIO_BYTES",
    "MAX_TRANSCRIPT_BYTES",
    "TranscriptionProvider",
    "TranscriptionResult",
    "UnavailableTranscriptionProvider",
    "validate_audio_input",
]
