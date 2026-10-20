"""Single application entry point; upstream route/policy/evidence remain authoritative."""
from dataclasses import dataclass
from config.runtime import RuntimeConfig
from delivery.pipeline import run_guarded_response
from delivery.fallbacks import FallbackRegistry
from semantic.checker import SemanticChecker, SemanticConfig
from semantic.contract import output_schema as semantic_schema
from evidence.checker import EvidenceChecker, output_schema as evidence_schema
from providers.http_transport import ChatCompletionsTransport, HTTPConfig

@dataclass(frozen=True)
class ProviderSpec:
    http: HTTPConfig
    model_revision: str
    deployment_approved: bool = False

    def __post_init__(self):
        if type(self.http) is not HTTPConfig:
            raise ValueError('Invalid provider specification')
        self.model_config(10.0)

    def model_config(self, timeout):
        return SemanticConfig(self.http.model_id, self.model_revision, self.deployment_approved, timeout)

class GuardrailService:
    def __init__(self, semantic=None, factuality=None, runtime=None, fallback_registry=None,
            rewriter=None, approved_templates=None):
        self.runtime = RuntimeConfig() if runtime is None else runtime
        self.registry = FallbackRegistry() if fallback_registry is None else fallback_registry
        self.rewriter, self.approved_templates = rewriter, approved_templates
        self.semantic = None if semantic is None else SemanticChecker(
            ChatCompletionsTransport(semantic.http, semantic_schema(), 'semantic_assessment'),
            semantic.model_config(self.runtime.semantic_checker_timeout_seconds))
        self.factuality = None if factuality is None else EvidenceChecker(
            ChatCompletionsTransport(factuality.http, evidence_schema(), 'evidence_assessment'),
            factuality.model_config(self.runtime.evidence_checker_timeout_seconds))

    def process(self, frozen_input):
        """Application renders only delivered_text; never generated_response on BLOCK."""
        return run_guarded_response(frozen_input, self.runtime, semantic_checker=self.semantic,
            evidence_checker=self.factuality, rewriter=self.rewriter, fallback_registry=self.registry,
            approved_templates=self.approved_templates)
