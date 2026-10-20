"""Lazy adapters for the frozen crisis and V11/V12 boundary components."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
import threading
from typing import Any, Mapping, Protocol

from .schemas import BoundaryDecision, ContractError, CrisisDecision


class CrisisGuardrail(Protocol):
    def assess(self, latest_user_message: str) -> Mapping[str, Any]: ...


class BoundaryGuardrail(Protocol):
    def predict(self, latest_user_message: str) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class ComponentStatus:
    name: str
    ready: bool
    version: str
    detail: str | None = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ready": self.ready,
            "version": self.version,
            "detail": self.detail,
        }


class LazyCrisisGuardrailAdapter:
    """Load the accepted crisis-v3 checkpoint only on first inference."""

    version = "crisis-input-guardrail/1.0"

    def __init__(
        self,
        *,
        component_root: str | Path,
        checkpoint: str | Path,
        device: str = "cpu",
        expected_model_sha256: str | None = None,
    ) -> None:
        self.component_root = Path(component_root).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.device = device
        self.expected_model_sha256 = expected_model_sha256.lower() if expected_model_sha256 else None
        self._guardrail: Any | None = None
        self._lock = threading.Lock()

    def readiness(self, *, verify_hash: bool = False) -> ComponentStatus:
        required = (
            self.checkpoint / "config.json",
            self.checkpoint / "calibration.json",
            self.checkpoint / "model.safetensors",
        )
        if not self.component_root.exists():
            return ComponentStatus("crisis_guardrail", False, self.version, "component_root_missing")
        if any(not path.is_file() for path in required):
            return ComponentStatus("crisis_guardrail", False, self.version, "checkpoint_incomplete")
        if verify_hash and self.expected_model_sha256:
            actual = sha256_file(self.checkpoint / "model.safetensors")
            if actual != self.expected_model_sha256:
                return ComponentStatus("crisis_guardrail", False, self.version, "model_hash_mismatch")
        return ComponentStatus("crisis_guardrail", True, self.version)

    def _load(self) -> Any:
        if self._guardrail is not None:
            return self._guardrail
        with self._lock:
            if self._guardrail is not None:
                return self._guardrail
            status = self.readiness(verify_hash=True)
            if not status.ready:
                raise RuntimeError(f"crisis_guardrail_not_ready:{status.detail}")
            _activate_component_root(self.component_root)
            classifier_module = importlib.import_module("src.input_guardrail.classifier")
            pipeline_module = importlib.import_module("src.input_guardrail.pipeline")
            classifier = classifier_module.CrisisV3Classifier(
                self.checkpoint,
                device=self.device,
            )
            self._guardrail = pipeline_module.InputGuardrail(classifier)
        return self._guardrail

    def assess(self, latest_user_message: str) -> Mapping[str, Any]:
        decision = self._load().assess(latest_user_message)
        result = decision.to_dict()
        if not isinstance(result, Mapping):
            raise ContractError("crisis adapter received a non-mapping result")
        return result


class LazyStaticCrisisAdapter(LazyCrisisGuardrailAdapter):
    """ST01c inference with an explicit bridge to the application policy schema.

    The frozen model prediction is retained intact under model_output. Timing
    and severity are not inferred by this bridge; unspecified/none are policy
    defaults and must never be presented as model assessments.
    """

    version = "crisis-static-v2/st01c-seed42-update880"

    def readiness(self, *, verify_hash: bool = False) -> ComponentStatus:
        status = super().readiness(verify_hash=verify_hash)
        if not status.ready:
            return status
        for name in ("static_heads.safetensors", "crisis_static_model_config.json"):
            if not (self.checkpoint / name).is_file():
                return ComponentStatus("crisis_guardrail", False, self.version, "checkpoint_incomplete")
        if verify_hash:
            manifest = json.loads((self.component_root / "MANIFEST.json").read_text(encoding="utf-8"))
            for name, metadata in manifest["files"].items():
                target = (self.component_root / name).resolve()
                target.relative_to(self.component_root)
                if not target.is_file() or sha256_file(target) != metadata["sha256"]:
                    return ComponentStatus("crisis_guardrail", False, self.version, "bundle_hash_mismatch")
        return status

    def _load(self) -> Any:
        if self._guardrail is not None:
            return self._guardrail
        with self._lock:
            if self._guardrail is None:
                status = self.readiness(verify_hash=True)
                if not status.ready:
                    raise RuntimeError(f"crisis_guardrail_not_ready:{status.detail}")
                root_text = str(self.component_root)
                if root_text not in sys.path:
                    sys.path.insert(0, root_text)
                module = importlib.import_module("crisis_classifier")
                origin = Path(module.__file__).resolve()
                origin.relative_to(self.component_root)
                self._guardrail = module.CrisisClassifier(self.checkpoint, device=self.device)
        return self._guardrail

    def assess(self, latest_user_message: str) -> Mapping[str, Any]:
        return bridge_static_crisis(self._load().predict(latest_user_message))


def bridge_static_crisis(prediction: Mapping[str, Any]) -> Mapping[str, Any]:
    """Preserve category overrides and presence-only crisis signals exactly."""
    if prediction.get("schema_version") != "crisis-static-v2":
        raise ContractError("unexpected ST01c prediction schema")
    action = prediction.get("action")
    tiers = {"crisis_specific_route": "crisis", "standard_response_eligible": "normal",
             "safety_check_or_uncertain": "monitored"}
    if action not in tiers:
        raise ContractError("unexpected ST01c action")
    labels = prediction.get("detected_categories")
    allowed = {"suicidal_ideation", "self_directed_harm", "sexual_violence",
               "domestic_violence", "child_abuse_or_endangerment"}
    if not isinstance(labels, list) or any(label not in allowed for label in labels):
        raise ContractError("unexpected ST01c categories")
    truncated = bool(prediction.get("truncated", False))
    # Preserve a detected crisis on an incomplete window. A negative truncated
    # result cannot authorize normal generation for the unseen suffix.
    tier = tiers[action]
    if truncated and tier == "normal":
        tier = "monitored"
    return {
        "tier": tier, "primary_category": labels[0] if labels else "no_crisis",
        "categories": list(labels), "temporal_status": "unspecified",
        "severity": "watch" if tier == "monitored" else "none", "topic_tags": [],
        "classifier": {"status": "input_truncated" if truncated else "ok",
                       "predicted_labels": list(labels), "near_threshold_labels": [],
                       "truncated": truncated},
        "rule_trace": {"rule_channel": {"matches": []}},
        "model_output": dict(prediction),
        "policy_bridge": {"timing_assessed": False, "severity_assessed": False,
                          "effective_action": "safety_check_or_uncertain" if truncated and action == "standard_response_eligible" else action},
    }


class LazyBoundaryV11Adapter:
    """Expose current V11 decisions through the V12-compatible shadow runtime."""

    version = "boundary-v11-v12r3-adapter/1.0"

    def __init__(
        self,
        *,
        component_root: str | Path,
        config_path: str | Path,
        device: str = "cpu",
    ) -> None:
        self.component_root = Path(component_root).expanduser().resolve()
        self.config_path = Path(config_path).expanduser().resolve()
        self.device = device
        self._runtime: Any | None = None
        self._lock = threading.Lock()

    def readiness(self) -> ComponentStatus:
        if not self.component_root.exists():
            return ComponentStatus("boundary_guardrail", False, self.version, "component_root_missing")
        if not self.config_path.is_file():
            return ComponentStatus("boundary_guardrail", False, self.version, "config_missing")
        return ComponentStatus("boundary_guardrail", True, self.version)

    def _load(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        with self._lock:
            if self._runtime is not None:
                return self._runtime
            status = self.readiness()
            if not status.ready:
                raise RuntimeError(f"boundary_guardrail_not_ready:{status.detail}")
            _activate_component_root(self.component_root)
            boundary_module = importlib.import_module("src.boundary_guardrail.v11_boundary")
            runtime_module = importlib.import_module("src.boundary_guardrail.v11_runtime")
            guard = boundary_module.V11BoundaryResearchGuard.from_config(
                self.config_path,
                device=self.device,
            )
            self._runtime = runtime_module.V11ShadowRuntime(guard)
        return self._runtime

    def predict(self, latest_user_message: str) -> Mapping[str, Any]:
        result = self._load().predict(latest_user_message)
        if not isinstance(result, Mapping):
            raise ContractError("boundary adapter received a non-mapping result")
        return result


class ParallelInputGuardrails:
    """Execute independent input channels concurrently with fail-closed timeouts."""

    version = "parallel-input-guardrails/1.0"

    def __init__(
        self,
        crisis_guardrail: CrisisGuardrail,
        boundary_guardrail: BoundaryGuardrail,
        *,
        timeout_seconds: float = 20.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.crisis_guardrail = crisis_guardrail
        self.boundary_guardrail = boundary_guardrail
        self.timeout_seconds = float(timeout_seconds)

    def assess(self, latest_user_message: str) -> tuple[CrisisDecision, BoundaryDecision]:
        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="input-safety")
        futures: dict[str, Future[Any]] = {
            "crisis": executor.submit(self.crisis_guardrail.assess, latest_user_message),
            "boundary": executor.submit(self.boundary_guardrail.predict, latest_user_message),
        }
        done, pending = wait(futures.values(), timeout=self.timeout_seconds)
        for future in pending:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)

        crisis = self._crisis_result(futures["crisis"], done)
        boundary = self._boundary_result(futures["boundary"], done)
        return crisis, boundary

    @staticmethod
    def _crisis_result(future: Future[Any], done: set[Future[Any]]) -> CrisisDecision:
        if future not in done:
            return CrisisDecision.fail_closed("timeout")
        try:
            return CrisisDecision.from_mapping(future.result())
        except Exception:
            return CrisisDecision.fail_closed("component_error")

    @staticmethod
    def _boundary_result(future: Future[Any], done: set[Future[Any]]) -> BoundaryDecision:
        if future not in done:
            return BoundaryDecision.fail_closed("timeout")
        try:
            return BoundaryDecision.from_mapping(future.result())
        except Exception:
            return BoundaryDecision.fail_closed("component_error")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


_ACTIVE_COMPONENT_ROOT: Path | None = None
_IMPORT_LOCK = threading.Lock()


def _activate_component_root(root: Path) -> None:
    """Make the legacy component's ``src`` package importable exactly once."""
    global _ACTIVE_COMPONENT_ROOT
    with _IMPORT_LOCK:
        if _ACTIVE_COMPONENT_ROOT is not None and _ACTIVE_COMPONENT_ROOT != root:
            raise RuntimeError("Only one guardrail component root may be active per process")
        root_text = str(root)
        if root_text not in sys.path:
            sys.path.insert(0, root_text)
        _ACTIVE_COMPONENT_ROOT = root
