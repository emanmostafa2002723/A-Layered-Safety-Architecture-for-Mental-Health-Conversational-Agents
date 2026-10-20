"""Construct the full pipeline from validated environment configuration."""

from __future__ import annotations

from pathlib import Path

from .config import ConfigurationError, PipelineConfig, RuntimeMode
from .generation import GenerationConfig, MistralGenerator, PromptBuilder, SYSTEM_PROMPT_PROFILES
from .guardrails import (
    LazyBoundaryV11Adapter,
    LazyStaticCrisisAdapter,
    ParallelInputGuardrails,
)
from .orchestrator import FullPipeline, PipelineComponents
from .output_guard import PlaceholderOutputGuard
from .rag import BM25Mode, HybridRAGBackend, HybridRAGConfig, RAGService
from .phoenix import PhoenixRAGBackend
from .qwen_router import QwenJointRouter
from .resources import ResourceDirectory
from .response_policy import ResponsePolicyBuilder
from .boundary_policy import IntegratedPolicyBuilder
from .new_components import S600CrisisAdapter, FinalBoundaryAdapter
from .fresh_crisis import FreshCrisisAdapter
from .output_integration import StandaloneOutputGate
from .local_output import LocalQwenOutputGate,JudgeSettings
from .hybrid_output import HybridOutputGate
from .qwen_only_output import QwenOnlyOutputGate
from .session_store import InMemorySafetyStateStore
from .worker_client import (
    JsonLineWorker,
    SubprocessBoundaryGuardrail,
    SubprocessCrisisGuardrail,
    WorkerConfig,
)


ACCEPTED_CRISIS_MODEL_SHA256 = (
    "d0844a55b3707257bef6f2eff628a148609bcd389103f180c796eb8ed763225a"
)


def build_pipeline(config: PipelineConfig) -> FullPipeline:
    """Build lazy components; no model weight is loaded during construction."""

    _require_paths(config)
    worker_environment = {
        "HF_HOME": str(config.model_cache),
        "HF_HUB_CACHE": str(Path(config.model_cache) / "hub"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    crisis_arguments = (
        "--component-root", str(config.crisis_checkpoint.parent),
        "--checkpoint", str(config.crisis_checkpoint),
        "--device", config.crisis_device,
        "--expected-model-sha256", ACCEPTED_CRISIS_MODEL_SHA256,
        "--weights", str(config.crisis_weights),
        "--weights-sha256", str(config.crisis_weights_sha256),
        "--backend", config.crisis_backend,
    )
    if config.crisis_backend == "fresh_epoch3":
        crisis_arguments += ("--calibration", str(config.crisis_calibration),
                             "--calibration-sha256", str(config.crisis_calibration_sha256))
    if config.guardrail_execution_mode == "subprocess":
        crisis_worker = JsonLineWorker(WorkerConfig(
            component="crisis",
            python_executable=config.crisis_python,  # type: ignore[arg-type]
            module="mental_health_pipeline.workers.crisis",
            arguments=crisis_arguments,
            source_root=config.project_root / "src",
            startup_timeout_seconds=config.worker_startup_timeout_seconds,
            request_timeout_seconds=config.worker_request_timeout_seconds,
            environment=worker_environment,
        ))
        boundary_worker = JsonLineWorker(WorkerConfig(
            component="boundary",
            python_executable=config.boundary_python,  # type: ignore[arg-type]
            module="mental_health_pipeline.workers.boundary_final",
            arguments=(
                "--component-root", str(config.boundary_root),
                "--device", config.boundary_device,
            ),
            source_root=config.project_root / "src",
            startup_timeout_seconds=config.worker_startup_timeout_seconds,
            request_timeout_seconds=config.worker_request_timeout_seconds,
            environment=worker_environment,
        ))
        crisis = SubprocessCrisisGuardrail(crisis_worker)
        boundary = SubprocessBoundaryGuardrail(boundary_worker)
    else:
        crisis = make_crisis_adapter(config)
        boundary = FinalBoundaryAdapter(config.boundary_root, config.boundary_device)

    input_guardrails = ParallelInputGuardrails(
        crisis, boundary, timeout_seconds=config.input_timeout_seconds
    )

    rag = None
    if config.enable_rag:
        if config.rag_router_backend not in {"legacy", "qwen_v4", "qwen_one_field", "minicpm_v1", "minicpm_v2"}:
            raise ConfigurationError("Unsupported RAG router backend")
        router = build_qwen_router(config) if config.rag_router_backend == "qwen_v4" else None
        if config.rag_router_backend.startswith('minicpm_') or config.rag_router_backend=='qwen_one_field':
            from .minicpm_router import build_minicpm_router
            router=build_minicpm_router(config)
        processed = Path(config.rag_project) / "data" / "processed"  # type: ignore[arg-type]
        rag = RAGService(PhoenixRAGBackend(config.rag_project, HybridRAGConfig(
            processed_dir=processed,
            embedding_model=config.embedding_model,
            reranker_model=config.reranker_model,
            embedding_device=config.retriever_device,
            reranker_device=config.reranker_device,
            allow_network=not config.offline_models,
            enable_dense=config.rag_enable_dense,
            enable_reranker=config.rag_enable_reranker,
            bm25_mode=BM25Mode.REBUILD_FROM_JSON,
        ), router=router))

    generator = None
    if config.enable_generation:
        if config.generator_prompt_profile not in SYSTEM_PROMPT_PROFILES:
            raise ConfigurationError('Unsupported generation prompt profile')
        generator = MistralGenerator(GenerationConfig(
            base_model=config.base_model,
            tokenizer=config.mistral_adapter,
            lora_adapter_path=config.mistral_adapter,
            device=config.generator_device,
            allow_network=not config.offline_models,
            quantization=config.generator_quantization,
            compute_dtype=config.generator_compute_dtype,
            gpu_memory_limit_mib=config.generator_gpu_memory_limit_mib,
            cpu_memory_limit_mib=config.generator_cpu_memory_limit_mib,
            offload_folder=config.generator_offload_folder,
            adapter_sha256=config.mistral_adapter_sha256,
            max_input_tokens=config.max_prompt_tokens,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            evidence_token_budget=760,
        ), prompt_builder=PromptBuilder(SYSTEM_PROMPT_PROFILES[config.generator_prompt_profile]))

    final_gate = None
    if config.output_guard_backend == "standalone":
        final_gate = StandaloneOutputGate(JsonLineWorker(WorkerConfig(
            component="output", python_executable=config.output_guard_python,
            module="mental_health_pipeline.workers.output",
            arguments=("--component-root", str(config.output_guard_root),
                       "--provider-config", str(config.output_provider_config)),
            source_root=config.project_root / "src", startup_timeout_seconds=60,
            request_timeout_seconds=45)), config.project_root / "config/governed_sources.json")
    elif config.output_guard_backend in {'local_qwen','local_hybrid'}:
        if config.runtime_mode is RuntimeMode.PRODUCTION:
            raise ConfigurationError('Local development output judge has no production acceptance')
        settings=JudgeSettings.load(config.local_output_settings)
        gate_type=HybridOutputGate if config.output_guard_backend=='local_hybrid' else QwenOnlyOutputGate
        extra_args=('--hybrid-project-root',str(config.project_root)) if config.output_guard_backend=='local_hybrid' else ()
        final_gate=gate_type(JsonLineWorker(WorkerConfig(
            component='output',python_executable=config.output_guard_python,
            module='mental_health_pipeline.workers.local_output',
            arguments=('--component-root',str(config.output_guard_root),
                       '--settings',str(config.local_output_settings))+extra_args,
            source_root=config.project_root/'src',startup_timeout_seconds=config.worker_startup_timeout_seconds,
            request_timeout_seconds=settings.timeout_seconds+10)))
    elif config.output_guard_backend != "placeholder":
        raise ConfigurationError("Unsupported output guard backend")
    resources = ResourceDirectory(
        config.resource_registry,
        production=config.runtime_mode is RuntimeMode.PRODUCTION,
    )
    components = PipelineComponents(
        input_guardrails=input_guardrails,
        response_policy=IntegratedPolicyBuilder(),
        resources=resources,
        state_store=InMemorySafetyStateStore(),
        output_guard=PlaceholderOutputGuard(),
        final_gate=final_gate,
        rag=rag,
        generator=generator,
    )
    return FullPipeline(config, components)


def make_crisis_adapter(config):
    if config.crisis_backend == 'fresh_epoch3':
        return FreshCrisisAdapter(component_root=config.crisis_weights.parent,
            weights=config.crisis_weights, calibration=config.crisis_calibration,
            weights_sha256=config.crisis_weights_sha256,
            calibration_sha256=config.crisis_calibration_sha256, device=config.crisis_device)
    if config.crisis_backend == 's600':
        return S600CrisisAdapter(
            weights=config.crisis_weights, weights_sha256=config.crisis_weights_sha256,
            component_root=config.crisis_checkpoint.parent,  # type: ignore[union-attr]
            checkpoint=config.crisis_checkpoint,  # type: ignore[arg-type]
            device=config.crisis_device,
            expected_model_sha256=ACCEPTED_CRISIS_MODEL_SHA256,
        )
    raise ConfigurationError('Unsupported crisis backend')


def build_qwen_router(config: PipelineConfig) -> QwenJointRouter:
    # Identical placement/context avoids reloading the same shared model on
    # every alternation between routing and judging. The frozen RAG prompt
    # and schema are unchanged.
    context_tokens=(JudgeSettings.load(config.local_output_settings).context_tokens
                    if config.output_guard_backend in {'local_qwen','local_hybrid'} else 4096)
    return QwenJointRouter(
        Path(config.rag_project) / "upstream/qwen_v4_2026-10-16/SYSTEM_PROMPT_V4.txt",
        api_url=config.qwen_api_url, model=config.qwen_model,
        timeout_seconds=config.qwen_timeout_seconds, num_gpu=config.qwen_num_gpu,
        context_tokens=context_tokens,
    )


def _require_paths(config: PipelineConfig) -> None:
    outside = [issue for issue in config.readiness_issues() if issue.endswith('_outside_project')]
    if outside:
        raise ConfigurationError('External runtime paths are not allowed in this copy: ' + ', '.join(outside))
    required = {
        "MHP_GUARDRAIL_REPO": config.guardrail_repo,
        "MHP_CRISIS_CHECKPOINT": config.crisis_checkpoint,
        "MHP_BOUNDARY_CONFIG": config.boundary_config,
        "MHP_MODEL_CACHE": config.model_cache,
    }
    if config.enable_rag:
        required["MHP_RAG_PROJECT"] = config.rag_project
    missing = [name for name, path in required.items() if path is None]
    if config.guardrail_execution_mode == "subprocess":
        if config.crisis_python is None:
            missing.append("MHP_CRISIS_PYTHON")
        if config.boundary_python is None:
            missing.append("MHP_BOUNDARY_PYTHON")
    if missing:
        raise ConfigurationError("Missing required configuration: " + ", ".join(missing))


__all__ = ["ACCEPTED_CRISIS_MODEL_SHA256", "build_pipeline"]
