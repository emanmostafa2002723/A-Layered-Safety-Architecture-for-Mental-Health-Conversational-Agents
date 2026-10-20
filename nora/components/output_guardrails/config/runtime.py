from dataclasses import dataclass
import math
from config.routes import SUPPORTED_ROUTES

@dataclass(frozen=True)
class RuntimeConfig:
    max_rewrite_attempts: int = 1
    controlled_rewrite_available: bool = False
    approved_fallback_routes: frozenset = frozenset()
    checker_timeout_seconds: float = 0.5
    semantic_checker_timeout_seconds: float = 10.0
    evidence_checker_timeout_seconds: float = 10.0
    request_timeout_seconds: float = 30.0

    def __post_init__(self):
        if type(self.max_rewrite_attempts) is not int or self.max_rewrite_attempts < 0:
            raise ValueError('Invalid rewrite limit')
        if type(self.controlled_rewrite_available) is not bool:
            raise ValueError('Invalid rewrite availability')
        if not isinstance(self.approved_fallback_routes, frozenset) or not self.approved_fallback_routes <= SUPPORTED_ROUTES:
            raise ValueError('Invalid approved fallback routes')
        for name in ('checker_timeout_seconds', 'semantic_checker_timeout_seconds', 'evidence_checker_timeout_seconds', 'request_timeout_seconds'):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid checker deadline: ' + name)
