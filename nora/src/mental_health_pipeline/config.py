"""Environment-driven configuration with startup validation and no secret logging."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from typing import Mapping

S600_WEIGHTS_SHA256 = "dcf1da8a409653b84a896588d1af8215f751b3abb6a0102880d0783c741f2b73"
PHOENIX_V5_SHA256 = "55d9b836b8a571fa57acb4d0e5bfd5f30b0ec63cd7954a38774ddb21c125fbc2"
PHOENIX_V6_SHA256 = "7742bb3438fb53c34862d88c031d2f8e5da2f0ceaff52c693804d570a9769da0"
FRESH_WEIGHTS_SHA256 = "82bb3befdebd3311833a14b299a921cbd120764f6b9dd75766fd61112c61bf68"
FRESH_CALIBRATION_SHA256 = "969101aee690025edfec12344b5f5980c3fb66a1b863ec50d87e4c4de9d7493d"


class RuntimeMode(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    runtime_mode: RuntimeMode = RuntimeMode.DEVELOPMENT
    guardrail_repo: Path | None = None
    crisis_checkpoint: Path | None = None
    crisis_weights: Path | None = None
    crisis_weights_sha256: str | None = FRESH_WEIGHTS_SHA256
    crisis_backend: str = "fresh_epoch3"
    crisis_calibration: Path | None = None
    crisis_calibration_sha256: str | None = FRESH_CALIBRATION_SHA256
    boundary_root: Path | None = None
    output_guard_root: Path | None = None
    output_guard_python: Path | None = None
    output_provider_config: Path | None = None
    local_output_settings: Path | None = None
    boundary_config: Path | None = None
    boundary_release: Path | None = None
    boundary_clinical_authorized: bool = False
    crisis_python: Path | None = None
    boundary_python: Path | None = None
    guardrail_execution_mode: str = "subprocess"
    rag_project: Path | None = None
    mistral_adapter: Path | None = None
    base_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "BAAI/bge-reranker-large"
    model_cache: Path | None = None
    crisis_device: str = "cpu"
    boundary_device: str = "cpu"
    generator_device: str = "cuda:0"
    generator_quantization: str | None = "4bit"
    generator_compute_dtype: str = "float16"
    generator_prompt_profile: str = "standard"
    generator_gpu_memory_limit_mib: int | None = None
    generator_cpu_memory_limit_mib: int | None = None
    generator_offload_folder: Path | None = None
    mistral_adapter_sha256: str | None = PHOENIX_V6_SHA256
    retriever_device: str = "cpu"
    reranker_device: str = "cpu"
    rag_enable_dense: bool = True
    rag_enable_reranker: bool = True
    rag_router_backend: str = "legacy"
    qwen_api_url: str = "http://127.0.0.1:11434/api/chat"
    qwen_model: str = "qwen3:4b"
    qwen_timeout_seconds: float = 30.0
    qwen_num_gpu: int = 0
    offline_models: bool = True
    input_timeout_seconds: float = 20.0
    worker_startup_timeout_seconds: float = 180.0
    worker_request_timeout_seconds: float = 30.0
    generation_timeout_seconds: float = 90.0
    max_new_tokens: int = 512
    max_prompt_tokens: int = 4096
    enable_generation: bool = True
    enable_rag: bool = True
    experimental_parallel_rag: bool = False
    experimental_incremental_output: bool = False
    enable_transcription: bool = False
    transcription_backend: str = "unavailable"
    output_guard_backend: str = "placeholder"
    gateway_token: str | None = field(default=None, repr=False)
    resource_registry: Path | None = None
    log_level: str = "INFO"
    extra: Mapping[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_env(
        cls,
        *,
        project_root: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "PipelineConfig":
        env = dict(os.environ if environ is None else environ)
        root = Path(project_root or env.get("MHP_PROJECT_ROOT") or Path.cwd()).expanduser().resolve()
        mode_text = env.get("MHP_RUNTIME_MODE", "development").strip().lower()
        try:
            mode = RuntimeMode(mode_text)
        except ValueError as error:
            raise ConfigurationError("MHP_RUNTIME_MODE must be development or production") from error

        def optional_path(
            name: str,
            default: Path | None = None,
            *,
            preserve_final_symlink: bool = False,
        ) -> Path | None:
            raw = env.get(name)
            if raw is None:
                path = default
            else:
                path = Path(raw).expanduser()
                if not path.is_absolute():
                    path = root / path
            if path is None:
                return None
            # POSIX virtual-environment executables are normally symlinks to the
            # base interpreter. Resolving the final component would turn
            # ``.venv-*/bin/python`` into (for example) ``/opt/conda/bin/python``
            # and make subprocess workers bypass the selected environment.
            if preserve_final_symlink:
                return path.parent.resolve() / path.name
            return path.resolve()

        rag_default = root / "rag_project"
        guardrail_default = root / "components" / "context_rules"
        crisis_default = root / "components" / "crisis_st01c" / "model"
        boundary_config_default = root / "components/boundary/calibration/candidate.json"
        boundary_release_default = root / "components/boundary"
        adapter_default = root / "models" / "phoenix_v6_safe_lora"
        base_model_default = root / "models" / "mistral-7b-instruct-v0.3"
        embedding_default = root / "models" / "all-MiniLM-L6-v2"
        reranker_default = root / "models" / "bge-reranker-large"
        model_cache_default = root / "models" / "huggingface"
        crisis_python_default = _project_venv_python(root, ".venv-crisis")
        boundary_python_default = _project_venv_python(root, ".venv-boundary")
        registry_default = root / "config" / "emergency_resources.v1.json"
        crisis_backend = env.get("MHP_CRISIS_BACKEND", "fresh_epoch3")
        crisis_weights_default = (root / "components/crisis_s600/final_weights.pt"
                                  if crisis_backend == "s600" else root / "components/crisis_fresh_epoch3/epoch_3.pt")
        crisis_hash_default = S600_WEIGHTS_SHA256 if crisis_backend == "s600" else FRESH_WEIGHTS_SHA256
        return cls(
            project_root=root,
            runtime_mode=mode,
            guardrail_repo=optional_path("MHP_GUARDRAIL_REPO", guardrail_default),
            crisis_checkpoint=optional_path("MHP_CRISIS_CHECKPOINT", crisis_default),
            crisis_weights=optional_path("MHP_CRISIS_WEIGHTS", crisis_weights_default),
            crisis_weights_sha256=env.get("MHP_CRISIS_WEIGHTS_SHA256", crisis_hash_default),
            crisis_backend=crisis_backend,
            crisis_calibration=optional_path("MHP_CRISIS_CALIBRATION", root / "components/crisis_fresh_epoch3/calibration.json"),
            crisis_calibration_sha256=env.get("MHP_CRISIS_CALIBRATION_SHA256", FRESH_CALIBRATION_SHA256),
            boundary_root=optional_path("MHP_BOUNDARY_ROOT", root / "components/boundary"),
            output_guard_root=optional_path("MHP_OUTPUT_GUARD_ROOT", root / "components/output_guardrails"),
            output_guard_python=optional_path("MHP_OUTPUT_GUARD_PYTHON", _project_venv_python(root,
                ".venv-api" if env.get('MHP_OUTPUT_GUARD_BACKEND') in {'local_hybrid','local_qwen'} else ".venv-output"), preserve_final_symlink=True),
            output_provider_config=optional_path("MHP_OUTPUT_PROVIDER_CONFIG", root / "config/output_providers.json"),
            local_output_settings=optional_path("MHP_LOCAL_OUTPUT_SETTINGS", root / "config" / (
                "local_output_hybrid.json" if env.get('MHP_OUTPUT_GUARD_BACKEND')=='local_hybrid' else "local_output_guard.json")),
            boundary_config=optional_path("MHP_BOUNDARY_CONFIG", boundary_config_default),
            boundary_release=optional_path("MHP_BOUNDARY_RELEASE", boundary_release_default),
            boundary_clinical_authorized=_bool(
                env.get("MHP_BOUNDARY_CLINICAL_AUTHORIZED", "false"),
                "MHP_BOUNDARY_CLINICAL_AUTHORIZED",
            ),
            crisis_python=optional_path(
                "MHP_CRISIS_PYTHON", crisis_python_default, preserve_final_symlink=True
            ),
            boundary_python=optional_path(
                "MHP_BOUNDARY_PYTHON", _project_venv_python(root, ".venv-api"), preserve_final_symlink=True
            ),
            guardrail_execution_mode=env.get(
                "MHP_GUARDRAIL_EXECUTION_MODE", "subprocess"
            ).strip().lower(),
            rag_project=optional_path("MHP_RAG_PROJECT", rag_default),
            mistral_adapter=optional_path("MHP_MISTRAL_ADAPTER", adapter_default),
            base_model=env.get("MHP_BASE_MODEL", str(base_model_default)),
            embedding_model=env.get("MHP_EMBEDDING_MODEL", str(embedding_default)),
            reranker_model=env.get("MHP_RERANKER_MODEL", str(reranker_default)),
            model_cache=optional_path("MHP_MODEL_CACHE", model_cache_default),
            crisis_device=env.get("MHP_CRISIS_DEVICE", "cpu"),
            boundary_device=env.get("MHP_BOUNDARY_DEVICE", "cpu"),
            generator_device=env.get("MHP_GENERATOR_DEVICE", "cuda:0"),
            generator_quantization=_optional_quantization(
                env.get("MHP_GENERATOR_QUANTIZATION", "4bit")
            ),
            generator_compute_dtype=env.get("MHP_GENERATOR_COMPUTE_DTYPE", "float16"),
            generator_prompt_profile=env.get('MHP_GENERATOR_PROMPT_PROFILE','standard'),
            generator_gpu_memory_limit_mib=_optional_positive_int(
                env.get("MHP_GENERATOR_GPU_MEMORY_LIMIT_MIB"),
                "MHP_GENERATOR_GPU_MEMORY_LIMIT_MIB",
            ),
            generator_cpu_memory_limit_mib=_optional_positive_int(
                env.get("MHP_GENERATOR_CPU_MEMORY_LIMIT_MIB"),
                "MHP_GENERATOR_CPU_MEMORY_LIMIT_MIB",
            ),
            generator_offload_folder=optional_path(
                "MHP_GENERATOR_OFFLOAD_FOLDER", root / ".runtime-offload"
            ),
            mistral_adapter_sha256=env.get("MHP_MISTRAL_ADAPTER_SHA256", PHOENIX_V6_SHA256),
            retriever_device=env.get("MHP_RETRIEVER_DEVICE", "cpu"),
            reranker_device=env.get("MHP_RERANKER_DEVICE", "cpu"),
            rag_enable_dense=_bool(
                env.get("MHP_RAG_ENABLE_DENSE", "true"), "MHP_RAG_ENABLE_DENSE"
            ),
            rag_enable_reranker=_bool(
                env.get("MHP_RAG_ENABLE_RERANKER", "true"),
                "MHP_RAG_ENABLE_RERANKER",
            ),
            rag_router_backend=env.get("MHP_RAG_ROUTER_BACKEND", "legacy").strip().lower(),
            qwen_api_url=env.get("MHP_OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat"),
            qwen_model=env.get("MHP_ROUTER_MODEL", "qwen3:4b"),
            qwen_timeout_seconds=_positive_float(
                env.get("MHP_ROUTER_TIMEOUT_SECONDS", "30"), "MHP_ROUTER_TIMEOUT_SECONDS"),
            qwen_num_gpu=_gpu_layers(env.get("MHP_QWEN_NUM_GPU", "0")),
            offline_models=_bool(env.get("MHP_OFFLINE_MODELS", "true"), "MHP_OFFLINE_MODELS"),
            input_timeout_seconds=_positive_float(env.get("MHP_INPUT_TIMEOUT_SECONDS", "20"), "MHP_INPUT_TIMEOUT_SECONDS"),
            worker_startup_timeout_seconds=_positive_float(
                env.get("MHP_WORKER_STARTUP_TIMEOUT_SECONDS", "180"),
                "MHP_WORKER_STARTUP_TIMEOUT_SECONDS",
            ),
            worker_request_timeout_seconds=_positive_float(
                env.get("MHP_WORKER_REQUEST_TIMEOUT_SECONDS", "30"),
                "MHP_WORKER_REQUEST_TIMEOUT_SECONDS",
            ),
            generation_timeout_seconds=_positive_float(env.get("MHP_GENERATION_TIMEOUT_SECONDS", "90"), "MHP_GENERATION_TIMEOUT_SECONDS"),
            max_new_tokens=_positive_int(env.get("MHP_MAX_NEW_TOKENS", "512"), "MHP_MAX_NEW_TOKENS"),
            max_prompt_tokens=_positive_int(env.get("MHP_MAX_PROMPT_TOKENS", "4096"), "MHP_MAX_PROMPT_TOKENS"),
            enable_generation=_bool(env.get("MHP_ENABLE_GENERATION", "true"), "MHP_ENABLE_GENERATION"),
            enable_rag=_bool(env.get("MHP_ENABLE_RAG", "true"), "MHP_ENABLE_RAG"),
            experimental_parallel_rag=_bool(env.get('MHP_EXPERIMENTAL_PARALLEL_RAG','false'), 'MHP_EXPERIMENTAL_PARALLEL_RAG'),
            experimental_incremental_output=_bool(env.get('MHP_EXPERIMENTAL_INCREMENTAL_OUTPUT','false'), 'MHP_EXPERIMENTAL_INCREMENTAL_OUTPUT'),
            enable_transcription=_bool(
                env.get("MHP_ENABLE_TRANSCRIPTION", "false"),
                "MHP_ENABLE_TRANSCRIPTION",
            ),
            transcription_backend=env.get(
                "MHP_TRANSCRIPTION_BACKEND", "unavailable"
            ).strip().lower(),
            output_guard_backend=env.get("MHP_OUTPUT_GUARD_BACKEND", "standalone"),
            gateway_token=env.get("MHP_GATEWAY_TOKEN") or None,
            resource_registry=optional_path("MHP_RESOURCE_REGISTRY", registry_default),
            log_level=env.get("MHP_LOG_LEVEL", "INFO").upper(),
        )

    def readiness_issues(self) -> tuple[str, ...]:
        checks = {
            "guardrail_repo_missing": self.guardrail_repo,
            "crisis_checkpoint_missing": self.crisis_checkpoint,
            "boundary_config_missing": self.boundary_config,
            "boundary_release_missing": self.boundary_release,
            "rag_project_missing": self.rag_project if self.enable_rag else self.project_root,
            "mistral_adapter_missing": self.mistral_adapter if self.enable_generation else self.project_root,
            "resource_registry_missing": self.resource_registry,
        }
        issues = [name for name, path in checks.items() if path is None or not path.exists()]
        if self.generator_prompt_profile not in {'standard','focused_v1','focused_v2','conditional_v3'}:
            issues.append('generator_prompt_profile_invalid')
        local_model_checks = {
            "base_model_missing": self.base_model if self.enable_generation else None,
            "embedding_model_missing": (
                self.embedding_model if self.enable_rag and self.rag_enable_dense else None
            ),
            "reranker_model_missing": (
                self.reranker_model if self.enable_rag and self.rag_enable_reranker else None
            ),
        }
        for code, configured in local_model_checks.items():
            if configured is None:
                continue
            candidate = Path(configured).expanduser()
            if candidate.is_absolute():
                if not candidate.exists():
                    issues.append(code)
            elif self.offline_models:
                issues.append(code.replace("_missing", "_must_be_local_in_offline_mode"))

        portable_paths: dict[str, Path | None] = {
            "guardrail_repo": self.guardrail_repo,
            "crisis_checkpoint": self.crisis_checkpoint,
            "boundary_config": self.boundary_config,
            "boundary_release": self.boundary_release,
            "crisis_python": self.crisis_python,
            "boundary_python": self.boundary_python,
            "rag_project": self.rag_project if self.enable_rag else None,
            "mistral_adapter": self.mistral_adapter if self.enable_generation else None,
            "model_cache": self.model_cache,
            "generator_offload_folder": self.generator_offload_folder,
            "resource_registry": self.resource_registry,
        }
        for name, path in portable_paths.items():
            within_project = (
                _is_executable_within(path, self.project_root)
                if path is not None and name in {"crisis_python", "boundary_python"}
                else path is None or _is_within(path, self.project_root)
            )
            if path is not None and not within_project:
                issues.append(f"{name}_outside_project")
        for name, path in (("crisis_weights", self.crisis_weights), ("boundary_root", self.boundary_root),
                           ("output_guard_root", self.output_guard_root), ("output_provider_config", self.output_provider_config)):
            if path is None or not path.exists():
                issues.append(name + "_missing")
            elif not _is_within(path, self.project_root):
                issues.append(name + "_outside_project")
        if not self.crisis_weights_sha256:
            issues.append("crisis_weights_hash_not_pinned")
        if self.crisis_backend not in {"fresh_epoch3", "s600"}:
            issues.append("crisis_backend_invalid")
        if self.crisis_backend == "s600":
            issues.append("crisis_s600_calibration_and_acceptance_pending")
        else:
            issues.append("crisis_fresh_independent_acceptance_pending")
            if self.crisis_calibration is None or not self.crisis_calibration.is_file():
                issues.append("crisis_calibration_missing")
            elif not _is_within(self.crisis_calibration, self.project_root):
                issues.append("crisis_calibration_outside_project")
            if not self.crisis_calibration_sha256:
                issues.append("crisis_calibration_hash_not_pinned")
        if self.output_guard_backend not in {"standalone", "placeholder", "local_qwen", "local_hybrid"}:
            issues.append("invalid_output_guard_backend")
        if self.output_guard_backend in {"standalone", "local_qwen", "local_hybrid"}:
            if self.output_guard_python is None or not self.output_guard_python.is_file():
                issues.append("output_guard_python_missing")
            elif not _is_executable_within(self.output_guard_python, self.project_root):
                issues.append("output_guard_python_outside_project")
        if self.output_guard_backend in {'local_qwen','local_hybrid'}:
            if self.local_output_settings is None or not self.local_output_settings.is_file():
                issues.append('local_output_settings_missing')
            elif not _is_within(self.local_output_settings,self.project_root):
                issues.append('local_output_settings_outside_project')
            if self.runtime_mode is RuntimeMode.PRODUCTION:
                issues.append('local_output_not_accepted_for_production')
        if self.runtime_mode is RuntimeMode.PRODUCTION:
            issues.append("production_candidate_not_accepted")
        for name, configured in (
            ("base_model", self.base_model if self.enable_generation else None),
            (
                "embedding_model",
                self.embedding_model if self.enable_rag and self.rag_enable_dense else None,
            ),
            (
                "reranker_model",
                self.reranker_model if self.enable_rag and self.rag_enable_reranker else None,
            ),
        ):
            if configured is None:
                continue
            candidate = Path(configured).expanduser()
            if candidate.is_absolute() and not _is_within(candidate, self.project_root):
                issues.append(f"{name}_outside_project")
        if self.runtime_mode is RuntimeMode.PRODUCTION and self.output_guard_backend == "placeholder":
            issues.append("production_output_guard_is_placeholder")
        if self.runtime_mode is RuntimeMode.PRODUCTION and not self.boundary_clinical_authorized:
            issues.append("production_boundary_clinical_authorization_missing")
        if self.runtime_mode is RuntimeMode.PRODUCTION and not self.gateway_token:
            issues.append("production_gateway_token_missing")
        if self.enable_transcription and self.transcription_backend == "unavailable":
            issues.append("transcription_backend_unavailable")
        if self.runtime_mode is RuntimeMode.PRODUCTION and not self.offline_models:
            issues.append("production_network_model_download_enabled")
        if self.generator_device.strip().lower() == "auto":
            issues.append("generator_device_must_be_explicit")
        if self.generator_quantization == "4bit" and self.generator_device.lower().startswith("cpu"):
            issues.append("four_bit_generator_requires_cuda")
        if self.guardrail_execution_mode not in {"subprocess", "in_process"}:
            issues.append("invalid_guardrail_execution_mode")
        if self.rag_router_backend not in {"legacy", "qwen_v4", "qwen_one_field", "minicpm_v1", "minicpm_v2"}:
            issues.append("invalid_rag_router_backend")
        if self.guardrail_execution_mode == "subprocess":
            for code, executable in (
                ("crisis_python_missing", self.crisis_python),
                ("boundary_python_missing", self.boundary_python),
            ):
                if executable is None or not executable.is_file():
                    issues.append(code)
        return tuple(issues)

    def require_production_ready(self) -> None:
        issues = self.readiness_issues()
        if issues:
            raise ConfigurationError("Production startup blocked: " + ", ".join(issues))


def _bool(value: str, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true or false")


def _positive_int(value: str, name: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be an integer") from error
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return parsed


def _positive_float(value: str, name: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ConfigurationError(f"{name} must be numeric") from error
    if parsed <= 0:
        raise ConfigurationError(f"{name} must be positive")
    return parsed


def _optional_quantization(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized in {"none", "off", "false", "0"}:
        return None
    if normalized == "4bit":
        return "4bit"
    raise ConfigurationError("MHP_GENERATOR_QUANTIZATION must be 4bit or none")


def _optional_positive_int(value: str | None, name: str) -> int | None:
    if value is None or value.strip().lower() in {"", "none", "off"}:
        return None
    return _positive_int(value, name)


def _gpu_layers(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise ConfigurationError("MHP_QWEN_NUM_GPU must be -1, 0, or positive") from error
    if result < -1:
        raise ConfigurationError("MHP_QWEN_NUM_GPU must be -1, 0, or positive")
    return result


def _project_venv_python(root: Path, directory: str) -> Path:
    windows = root / directory / "Scripts" / "python.exe"
    posix = root / directory / "bin" / "python"
    return windows if os.name == "nt" else posix


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _is_executable_within(path: Path, root: Path) -> bool:
    """Check the configured launcher location without dereferencing its final symlink."""
    try:
        path.parent.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True
