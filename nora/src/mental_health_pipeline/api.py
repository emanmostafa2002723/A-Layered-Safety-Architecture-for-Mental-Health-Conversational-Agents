"""FastAPI transport for the fully buffered mental-health pipeline.

The transport performs no clinical routing and does not run an output guard a
second time.  Those responsibilities belong to the orchestrator.  This layer
limits requests, constructs the versioned request contract, runs blocking model
work outside the event loop, requires a complete ``PipelineResponse``, and
serializes only its explicitly public representation.

FastAPI is an optional deployment dependency: importing this module remains
safe when it is not installed, while ``create_app`` fails with a clear error.
"""

from __future__ import annotations

import asyncio
import hmac
import inspect
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Protocol, runtime_checkable

from .resources import DEFAULT_RESOURCE_DIRECTORY, ResourceDirectory
from .schemas import (
    MAX_MESSAGE_BYTES,
    PipelineRequest,
    PipelineResponse,
    RequestScope,
    RequestValidationError,
)
from .telemetry import (
    NullTelemetry,
    TelemetryEvent,
    TelemetrySink,
    byte_length,
    error_code_for,
)


try:  # pragma: no branch - the two environments are tested separately
    from fastapi import FastAPI, Query, Request
    from fastapi.exceptions import RequestValidationError as FastAPIRequestValidationError
    from fastapi.responses import HTMLResponse, JSONResponse
    from pydantic import BaseModel, ConfigDict, Field

    FASTAPI_AVAILABLE = True
    _FASTAPI_IMPORT_ERROR: BaseException | None = None
except ImportError as error:  # FastAPI is intentionally optional for core/model workers
    FASTAPI_AVAILABLE = False
    _FASTAPI_IMPORT_ERROR = error
    FastAPI = Any  # type: ignore[misc,assignment]


API_VERSION = "1.0.0"
SERVICE_UNAVAILABLE_RESPONSE = (
    "I’m sorry, but the service is temporarily unavailable. If you or someone "
    "else may be in immediate danger, contact the local emergency number or go "
    "to the nearest emergency department now."
)


class FastAPIUnavailableError(RuntimeError):
    """Raised only when an application is requested without FastAPI installed."""


class APIConfigurationError(ValueError):
    pass


class BufferedResponseContractError(TypeError):
    pass


@dataclass(frozen=True, slots=True)
class APISettings:
    environment: str = "development"
    max_request_body_bytes: int = 20_000
    max_response_bytes: int = 128_000
    request_timeout_seconds: float = 150.0
    max_concurrent_requests: int = 1
    block_unready_production_startup: bool = True
    docs_enabled: bool = True
    chat_ui_enabled: bool = False
    gateway_token: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.environment not in {"development", "test", "production"}:
            raise APIConfigurationError("environment must be development, test, or production")
        if self.max_request_body_bytes < MAX_MESSAGE_BYTES:
            raise APIConfigurationError(
                f"max_request_body_bytes must be at least {MAX_MESSAGE_BYTES}"
            )
        if self.max_response_bytes < 1_024:
            raise APIConfigurationError("max_response_bytes must be at least 1024")
        if self.request_timeout_seconds <= 0:
            raise APIConfigurationError("request_timeout_seconds must be positive")
        if self.max_concurrent_requests < 1:
            raise APIConfigurationError("max_concurrent_requests must be positive")
        if self.environment == "production" and not self.gateway_token:
            raise APIConfigurationError("production requires a trusted gateway token")


@runtime_checkable
class PipelineOrchestrator(Protocol):
    """Minimal orchestrator dependency required by the HTTP transport."""

    def process(self, request: PipelineRequest) -> PipelineResponse: ...


if FASTAPI_AVAILABLE:

    class ChatRequestBody(BaseModel):
        model_config = ConfigDict(extra="forbid")

        message: str = Field(min_length=1, max_length=MAX_MESSAGE_BYTES)
        session_id: str = Field(min_length=1, max_length=128)
        turn_number: int = Field(default=1, ge=1)
        age_eligibility_18_plus: bool | None = None
        preferred_language: str = Field(default="en", min_length=1, max_length=32)
        current_location_country: str | None = Field(default=None, max_length=64)
        current_location_region: str | None = Field(default=None, max_length=128)
        location_confirmed: bool = False


    class CitationBody(BaseModel):
        model_config = ConfigDict(extra="forbid")

        source_id: str
        title: str
        source: str
        page: str | None = None
        url: str | None = None


    class ChatResponseBody(BaseModel):
        model_config = ConfigDict(extra="forbid")

        schema_version: str
        request_id: str
        response: str
        route: str
        citations: list[CitationBody]
        resources: dict[str, Any] | None
        used_fallback: bool

else:

    @dataclass(slots=True)
    class ChatRequestBody:  # type: ignore[no-redef]
        message: str
        session_id: str
        turn_number: int = 1
        age_eligibility_18_plus: bool | None = None
        preferred_language: str = "en"
        current_location_country: str | None = None
        current_location_region: str | None = None
        location_confirmed: bool = False


    @dataclass(slots=True)
    class CitationBody:  # type: ignore[no-redef]
        source_id: str
        title: str
        source: str
        page: str | None = None
        url: str | None = None


    @dataclass(slots=True)
    class ChatResponseBody:  # type: ignore[no-redef]
        schema_version: str
        request_id: str
        response: str
        route: str
        citations: list[CitationBody]
        resources: dict[str, Any] | None
        used_fallback: bool


class _RequestTooLarge(Exception):
    pass


class RequestBodyLimitMiddleware:
    """ASGI body limit that also covers chunked requests."""

    def __init__(self, app: Any, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", ())}
        raw_length = headers.get(b"content-length")
        if raw_length is not None:
            try:
                if int(raw_length) > self.max_bytes:
                    await _send_json(send, 413, {"detail": "request_body_too_large"})
                    return
            except ValueError:
                await _send_json(send, 400, {"detail": "invalid_content_length"})
                return

        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestTooLarge:
            await _send_json(send, 413, {"detail": "request_body_too_large"})


class SecurityHeadersMiddleware:
    """Prevent caching and MIME sniffing of sensitive chat responses."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        async def secure_send(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", ()))
                headers.extend(
                    (
                        (b"cache-control", b"no-store"),
                        (b"pragma", b"no-cache"),
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"no-referrer"),
                    )
                )
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, secure_send)


class GatewayAuthMiddleware:
    """Require a shared gateway credential for public API routes.

    Health checks remain available to the local orchestrator. Deployments must
    terminate TLS and enforce client rate limits at the trusted gateway; this
    token prevents direct access to the model service itself.
    """

    def __init__(self, app: Any, *, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = str(scope.get("path", ""))
        if scope.get("type") != "http" or not path.startswith("/v1/"):
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", ())}
        supplied = headers.get(b"x-api-key", b"").decode("utf-8", errors="ignore")
        authorization = headers.get(b"authorization", b"").decode(
            "utf-8", errors="ignore"
        )
        if authorization.casefold().startswith("bearer "):
            supplied = authorization[7:].strip()
        if not supplied or not hmac.compare_digest(supplied, self.token):
            await _send_json(send, 401, {"detail": "unauthorized"})
            return
        await self.app(scope, receive, send)


async def _send_json(send: Any, status: int, payload: Mapping[str, Any]) -> None:
    body = json.dumps(dict(payload), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def build_pipeline_request(body: ChatRequestBody, *, request_id: str) -> PipelineRequest:
    """Translate HTTP input into the sole application request contract."""

    request = PipelineRequest(
        latest_user_message=body.message,
        session_id=body.session_id,
        scope=RequestScope(
            age_eligibility_18_plus=body.age_eligibility_18_plus,
            preferred_language=body.preferred_language,
            current_location_country=body.current_location_country,
            current_location_region=body.current_location_region,
            location_confirmed=body.location_confirmed,
        ),
        turn_number=body.turn_number,
        request_id=request_id,
    )
    request.validate()
    return request


async def invoke_orchestrator(
    orchestrator: PipelineOrchestrator,
    request: PipelineRequest,
) -> PipelineResponse:
    """Run synchronous model work off-loop and require one complete response."""

    process = getattr(orchestrator, "process", None)
    if not callable(process):
        raise BufferedResponseContractError("orchestrator must expose process(request)")

    if inspect.iscoroutinefunction(process):
        result = await process(request)
    else:
        result = await asyncio.to_thread(process, request)
        if inspect.isawaitable(result):
            result = await result

    if not isinstance(result, PipelineResponse):
        raise BufferedResponseContractError(
            "orchestrator must return a complete PipelineResponse"
        )
    if not isinstance(result.response, str) or (not result.response.strip() and result.output_guard_status != "output_guard_abstained"):
        raise BufferedResponseContractError("PipelineResponse.response must be complete text")
    if result.request_id != request.request_id:
        raise BufferedResponseContractError("PipelineResponse.request_id does not match request")
    return result


async def probe_readiness(orchestrator: PipelineOrchestrator) -> dict[str, Any]:
    """Normalize an optional orchestrator readiness probe without guessing ready."""

    probe = getattr(orchestrator, "readiness", None)
    if callable(probe):
        if inspect.iscoroutinefunction(probe):
            result = await probe()
        else:
            result = await asyncio.to_thread(probe)
            if inspect.isawaitable(result):
                result = await result
        if isinstance(result, bool):
            return {"ready": result, "issues": [] if result else ["orchestrator_not_ready"]}
        if isinstance(result, Mapping):
            ready = bool(result.get("ready", False))
            issues = result.get("issues", ())
            if not isinstance(issues, (list, tuple)):
                issues = ("invalid_readiness_contract",)
                ready = False
            return {
                "ready": ready,
                "issues": [str(item)[:128] for item in issues],
            }

    static_ready = getattr(orchestrator, "production_ready", None)
    if isinstance(static_ready, bool):
        return {
            "ready": static_ready,
            "issues": [] if static_ready else ["orchestrator_not_production_ready"],
        }
    return {"ready": False, "issues": ["orchestrator_readiness_unavailable"]}


def create_app(
    orchestrator: PipelineOrchestrator,
    *,
    settings: APISettings | None = None,
    telemetry: TelemetrySink | None = None,
    resource_directory: ResourceDirectory | None = None,
) -> FastAPI:
    """Build an app with explicit, replaceable production dependencies."""

    if not FASTAPI_AVAILABLE:
        raise FastAPIUnavailableError(
            "FastAPI is required to create the HTTP service; install the deployment dependencies"
        ) from _FASTAPI_IMPORT_ERROR
    if not callable(getattr(orchestrator, "process", None)):
        raise APIConfigurationError("orchestrator must expose process(request)")

    api_settings = settings or APISettings()
    telemetry_sink = telemetry or NullTelemetry()
    resources = resource_directory or DEFAULT_RESOURCE_DIRECTORY
    capacity = asyncio.Semaphore(api_settings.max_concurrent_requests)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            if (
                api_settings.environment == "production"
                and api_settings.block_unready_production_startup
            ):
                readiness = await probe_readiness(orchestrator)
                if not readiness["ready"]:
                    raise RuntimeError("production_startup_blocked_not_ready")
            startup = getattr(orchestrator, "startup", None)
            if callable(startup):
                await asyncio.to_thread(startup)
            yield
        finally:
            close = getattr(orchestrator, "close", None)
            if callable(close):
                await asyncio.to_thread(close)

    docs_url = "/docs" if api_settings.docs_enabled else None
    redoc_url = "/redoc" if api_settings.docs_enabled else None
    app = FastAPI(
        title="Mental Health Support Pipeline API",
        version=API_VERSION,
        docs_url=docs_url,
        redoc_url=redoc_url,
        lifespan=lifespan,
    )
    app.state.orchestrator = orchestrator
    app.state.settings = api_settings
    app.state.telemetry = telemetry_sink
    app.state.resource_directory = resources
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=api_settings.max_request_body_bytes)
    if api_settings.gateway_token:
        app.add_middleware(GatewayAuthMiddleware, token=api_settings.gateway_token)

    if api_settings.chat_ui_enabled:
        from .web_ui import LOCAL_CHAT_HTML

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def local_chat_ui() -> HTMLResponse:
            return HTMLResponse(LOCAL_CHAT_HTML)

    @app.exception_handler(FastAPIRequestValidationError)
    async def validation_error_handler(
        _request: Request,
        _error: FastAPIRequestValidationError,
    ) -> JSONResponse:
        # Do not echo rejected request values from Pydantic's error context.
        return JSONResponse(status_code=422, content={"detail": "invalid_request"})

    @app.get("/health/live", include_in_schema=False)
    async def live() -> dict[str, str]:
        return {"status": "live", "api_version": API_VERSION}

    @app.get("/health/ready", include_in_schema=False)
    async def ready() -> JSONResponse:
        report = await probe_readiness(orchestrator)
        payload = {
            "status": "ready" if report["ready"] else "not_ready",
            "api_version": API_VERSION,
            "issues": report["issues"],
        }
        return JSONResponse(status_code=200 if report["ready"] else 503, content=payload)

    @app.get("/v1/resources", response_model=None)
    async def crisis_resources(
        country: str | None = Query(default=None, max_length=64),
        location_confirmed: bool = Query(default=False),
    ) -> dict[str, object]:
        # An unconfirmed country is deliberately ignored; location is never inferred.
        selected_country = country if location_confirmed else None
        return resources.resolve(selected_country).to_dict()

    @app.post("/v1/chat", response_model=ChatResponseBody)
    async def chat(body: ChatRequestBody) -> Any:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        input_size = 0

        try:
            pipeline_request = build_pipeline_request(body, request_id=request_id)
            input_size = byte_length(pipeline_request.latest_user_message)
            async with capacity:
                response = await asyncio.wait_for(
                    invoke_orchestrator(orchestrator, pipeline_request),
                    timeout=api_settings.request_timeout_seconds,
                )
            if response.output_guard_status == "output_guard_abstained":
                return JSONResponse(status_code=503, content={"detail": "output_guard_no_display",
                    "request_id": request_id}, headers={"X-Request-ID": request_id})
            output_size = byte_length(response.response)
            if output_size > api_settings.max_response_bytes:
                raise BufferedResponseContractError("PipelineResponse exceeds output limit")

            _record_safely(
                telemetry_sink,
                TelemetryEvent(
                    event="api.request_completed",
                    request_id=request_id,
                    outcome="success",
                    duration_ms=(time.perf_counter() - started) * 1_000,
                    input_bytes=input_size,
                    output_bytes=output_size,
                    route=response.route,
                    used_fallback=response.used_fallback,
                ),
            )
            public_payload = response.public_dict()
            return JSONResponse(
                status_code=200,
                content=public_payload,
                headers={"X-Request-ID": request_id},
            )
        except RequestValidationError:
            _record_safely(
                telemetry_sink,
                TelemetryEvent(
                    event="api.request_rejected",
                    request_id=request_id,
                    outcome="invalid_request",
                    duration_ms=(time.perf_counter() - started) * 1_000,
                    input_bytes=input_size,
                    output_bytes=0,
                    error_code="RequestValidationError",
                ),
            )
            return JSONResponse(
                status_code=422,
                content={"detail": "invalid_request", "request_id": request_id},
                headers={"X-Request-ID": request_id},
            )
        except Exception as error:
            # A transport error is not permission to bypass the selected guard
            # with the API's older static fallback/resource path.
            if getattr(getattr(orchestrator, "config", None), "output_guard_backend", None) in {"standalone","local_qwen","local_hybrid"}:
                return JSONResponse(status_code=503, content={"detail": "pipeline_no_checked_response",
                    "request_id": request_id}, headers={"X-Request-ID": request_id})
            # Deliberately do not log or return exception text.  The fallback does
            # not assume a country, but its attached resources may use an explicit,
            # confirmed request country if one was successfully parsed.
            confirmed_country: str | None = None
            try:
                if body.location_confirmed:
                    confirmed_country = body.current_location_country
            except (AttributeError, TypeError):
                pass
            resource_bundle = resources.resolve(confirmed_country)
            fallback_payload = {
                "schema_version": "pipeline-response/1.0",
                "request_id": request_id,
                "response": SERVICE_UNAVAILABLE_RESPONSE,
                "route": "service_unavailable",
                "citations": [],
                "resources": resource_bundle.to_dict(),
                "used_fallback": True,
            }
            _record_safely(
                telemetry_sink,
                TelemetryEvent(
                    event="api.request_failed",
                    request_id=request_id,
                    outcome="service_unavailable",
                    duration_ms=(time.perf_counter() - started) * 1_000,
                    input_bytes=input_size,
                    output_bytes=byte_length(SERVICE_UNAVAILABLE_RESPONSE),
                    route="service_unavailable",
                    used_fallback=True,
                    error_code=error_code_for(error),
                ),
            )
            return JSONResponse(
                status_code=503,
                content=fallback_payload,
                headers={"X-Request-ID": request_id},
            )

    return app


def _record_safely(sink: TelemetrySink, event: TelemetryEvent) -> None:
    try:
        sink.record(event)
    except Exception:
        # Observability must never expose content or take down the serving path.
        return


__all__ = [
    "APIConfigurationError",
    "APISettings",
    "API_VERSION",
    "BufferedResponseContractError",
    "ChatRequestBody",
    "ChatResponseBody",
    "FASTAPI_AVAILABLE",
    "FastAPIUnavailableError",
    "GatewayAuthMiddleware",
    "PipelineOrchestrator",
    "SERVICE_UNAVAILABLE_RESPONSE",
    "build_pipeline_request",
    "create_app",
    "invoke_orchestrator",
    "probe_readiness",
]
