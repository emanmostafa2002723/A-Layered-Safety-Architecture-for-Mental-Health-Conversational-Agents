"""Read-only deployment preflight checks; no model is loaded."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata, util
import json
from pathlib import Path
import platform
import shutil
from typing import Any, Iterable

from .artifacts import verify_manifest
from .config import PipelineConfig, RuntimeMode
from .resources import ResourceDirectory, ResourceRegistryError


MINIMUM_FREE_DISK_BYTES = 25 * 1024**3
RECOMMENDED_GENERATOR_VRAM_BYTES = 8 * 1024**3


@dataclass(frozen=True)
class ReadinessIssue:
    code: str
    severity: str
    component: str
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    ready: bool
    mode: str
    issues: tuple[ReadinessIssue, ...]
    environment: dict[str, Any]
    artifacts: tuple[dict[str, Any], ...]
    schema_version: str = "deployment-preflight/1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ready": self.ready,
            "mode": self.mode,
            "issues": [asdict(item) for item in self.issues],
            "environment": self.environment,
            "artifacts": list(self.artifacts),
        }


def run_preflight(config: PipelineConfig, *, manifest_path: str | Path | None = None) -> PreflightReport:
    issues: list[ReadinessIssue] = []
    blocker_severity = "blocker" if config.runtime_mode is RuntimeMode.PRODUCTION else "warning"
    for code in config.readiness_issues():
        severity = "blocker" if code.startswith("production_") else blocker_severity
        issues.append(ReadinessIssue(code, severity, "configuration", code.replace("_", " ")))

    disk = shutil.disk_usage(config.project_root)
    if disk.free < MINIMUM_FREE_DISK_BYTES:
        issues.append(ReadinessIssue(
            "insufficient_free_disk",
            "blocker" if config.runtime_mode is RuntimeMode.PRODUCTION else "warning",
            "host",
            f"At least 25 GiB free is required; detected {disk.free / 1024**3:.2f} GiB.",
        ))

    packages = _package_versions((
        "torch", "transformers", "peft", "sentence-transformers", "faiss-cpu",
        "rank-bm25", "fastapi", "uvicorn", "pydantic",
    ))
    required_modules = (
        "torch", "transformers", "peft", "pydantic", "fastapi", "uvicorn", "yaml"
    )
    for module_name in required_modules:
        if util.find_spec(module_name) is None:
            issues.append(ReadinessIssue(
                f"dependency_missing:{module_name}", blocker_severity, "dependencies",
                f"Required Python module {module_name!r} is not installed.",
            ))
    if config.enable_rag:
        rag_modules: list[str] = ["numpy", "rank_bm25"]
        if config.rag_enable_dense:
            rag_modules.extend(("sentence_transformers", "faiss"))
        if config.rag_enable_reranker and "sentence_transformers" not in rag_modules:
            rag_modules.append("sentence_transformers")
        for module_name in rag_modules:
            if util.find_spec(module_name) is None:
                issues.append(ReadinessIssue(
                    f"rag_dependency_missing:{module_name}", blocker_severity, "rag",
                    f"RAG module {module_name!r} is not installed.",
                ))
        if config.rag_router_backend != 'legacy':
            from .bootstrap import build_qwen_router
            try:
                if config.rag_router_backend=='qwen_v4':router=build_qwen_router(config)
                else:
                    from .minicpm_router import build_minicpm_router
                    router=build_minicpm_router(config)
                code = router.readiness_code()
            except (OSError,ValueError,KeyError,TypeError):
                code = "router_configuration_invalid"
            if code:
                issues.append(ReadinessIssue(code, blocker_severity, "qwen_router", code))

    artifact_rows: tuple[dict[str, Any], ...] = ()
    manifest = Path(manifest_path or config.project_root / "config" / "artifacts.json")
    if manifest.is_file():
        checks = verify_manifest(manifest, project_root=config.project_root)
        artifact_rows = tuple(item.public_dict() for item in checks)
        for item in checks:
            if not item.ready:
                issues.append(ReadinessIssue(
                    f"artifact_not_ready:{item.artifact_id}",
                    blocker_severity,
                    "artifacts",
                    "; ".join(item.issues),
                ))
    else:
        issues.append(ReadinessIssue(
            "artifact_manifest_missing", blocker_severity, "artifacts",
            "The hash-bound artifact manifest is missing.",
        ))

    if config.resource_registry and config.resource_registry.is_file():
        try:
            directory = ResourceDirectory(config.resource_registry)
            for resource_issue in directory.readiness_issues():
                issues.append(ReadinessIssue(
                    f"resource:{resource_issue}", blocker_severity, "resources",
                    resource_issue.replace("_", " "),
                ))
        except ResourceRegistryError:
            issues.append(ReadinessIssue(
                "resource_registry_invalid", blocker_severity, "resources",
                "The emergency-resource registry is invalid.",
            ))

    if config.boundary_release is not None:
        release_path = config.boundary_release / "RELEASE.json"
        try:
            release = json.loads(release_path.read_text(encoding="utf-8"))
            if not isinstance(release, dict) or release.get("clinical_deployment_ready") is not True:
                issues.append(ReadinessIssue(
                    "boundary_release_not_clinically_authorized",
                    blocker_severity,
                    "boundary_guardrail",
                    "The selected boundary release does not authorize clinical deployment.",
                ))
        except (OSError, UnicodeError, json.JSONDecodeError):
            issues.append(ReadinessIssue(
                "boundary_release_metadata_invalid", blocker_severity,
                "boundary_guardrail", "Boundary release metadata could not be verified.",
            ))

    if config.output_guard_backend == 'standalone':
        try:
            providers = json.loads(config.output_provider_config.read_text(encoding='utf-8'))
            configured = all(isinstance(providers.get(role), dict) and
                             providers[role].get('deployment_approved') is True
                             and all(providers[role].get(key) for key in ('endpoint', 'model_id', 'model_revision'))
                             for role in ('semantic', 'factuality'))
            configured = configured and providers.get('data_transmission_authorized') is True
        except (OSError, ValueError, AttributeError):
            configured = False
        if not configured:
            issues.append(ReadinessIssue('output_providers_missing_or_unapproved', 'blocker', 'output_guard',
                'Standalone gate is wired but will ABSTAIN. Obtain approved semantic/factuality providers.'))
    if config.output_guard_backend in {'local_qwen','local_hybrid'}:
        try:
            from .local_output import JudgeSettings
            JudgeSettings.load(config.local_output_settings)
            if config.output_guard_backend=='local_hybrid':
                from .hybrid_output import validate_medical_assets
                model=validate_medical_assets(config.local_output_settings,config.project_root)
                calibration=json.loads((model/'calibration.json').read_text(encoding='utf-8'))
                if not all(calibration['calibration_recall_target_met']):
                    issues.append(ReadinessIssue('medical_head_not_standalone_accepted',blocker_severity,'output_guard',
                        'Medical head misses diagnosis cases. Additional veto only; clear results require the complete Qwen/rules judge.'))
            issues.append(ReadinessIssue('local_output_development_only',blocker_severity,'output_guard',
                'Project-owned local Qwen safety/entailment judge selected. Clinical acceptance is pending.'))
        except (OSError,ValueError,TypeError,KeyError):
            issues.append(ReadinessIssue('local_output_configuration_invalid','blocker','output_guard',
                'Local output-judge settings could not be validated.'))
    accelerator = _accelerator_snapshot()
    if config.enable_generation and config.generator_device.lower().startswith("cuda"):
        devices = accelerator.get("devices", [])
        if not accelerator.get("cuda_available") or not devices:
            issues.append(ReadinessIssue(
                "generator_cuda_unavailable", blocker_severity, "generator",
                "Generation is assigned to CUDA but no CUDA device is available.",
            ))
        elif max(int(item.get("total_vram_bytes", 0)) for item in devices) < RECOMMENDED_GENERATOR_VRAM_BYTES:
            issues.append(ReadinessIssue(
                "generator_gpu_vram_below_recommended_minimum", blocker_severity,
                "generator",
                "A dedicated GPU with at least 8 GiB VRAM is recommended for the configured 7B 4-bit runtime.",
            ))
    environment = {
        "rag": {"router_backend": config.rag_router_backend,
                "dense_enabled": config.rag_enable_dense,
                "reranker_enabled": config.rag_enable_reranker,
                "qwen_num_gpu": config.qwen_num_gpu},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "disk_free_bytes": disk.free,
        "packages": packages,
        "accelerator": accelerator,
    }
    ready = not any(item.severity == "blocker" for item in issues)
    return PreflightReport(
        ready=ready,
        mode=config.runtime_mode.value,
        issues=tuple(issues),
        environment=environment,
        artifacts=artifact_rows,
    )


def _package_versions(distributions: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in distributions:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _accelerator_snapshot() -> dict[str, Any]:
    if util.find_spec("torch") is None:
        return {"torch_available": False, "cuda_available": False, "devices": []}
    try:
        import torch

        devices = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(index)
                devices.append({
                    "index": index,
                    "name": properties.name,
                    "total_vram_bytes": int(properties.total_memory),
                })
        return {
            "torch_available": True,
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_build_version": torch.version.cuda,
            "devices": devices,
        }
    except Exception:
        return {"torch_available": True, "cuda_available": False, "devices": [], "query_failed": True}


def write_report(report: PreflightReport, destination: str | Path) -> None:
    Path(destination).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
