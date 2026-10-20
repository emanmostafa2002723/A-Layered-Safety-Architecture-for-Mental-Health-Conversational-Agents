"""Deployment-oriented orchestration for the mental-health chat pipeline.

The package keeps safety classification, response policy, retrieval, generation,
and output validation as independent, versioned components.  Importing it never
loads model weights; model-backed adapters are lazy by design.
"""

from .schemas import (
    CombinedSafetyDecision,
    PipelineRequest,
    PipelineResponse,
    RequestScope,
    SystemRoute,
)

__all__ = [
    "CombinedSafetyDecision",
    "PipelineRequest",
    "PipelineResponse",
    "RequestScope",
    "SystemRoute",
]

__version__ = "0.3.0.dev20261018"
