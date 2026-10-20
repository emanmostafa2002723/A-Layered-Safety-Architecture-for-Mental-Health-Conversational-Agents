"""Explicit adapters for the 18 October integration candidate; no implicit promotion."""
from pathlib import Path
import threading

from .guardrails import LazyStaticCrisisAdapter, bridge_static_crisis, sha256_file


def load_weights_only(model, checkpoint, expected_hash):
    """Strict complete encoder+heads replacement, never pickle arbitrary objects."""
    import torch
    if not expected_hash or sha256_file(checkpoint) != expected_hash:
        raise ValueError("Checkpoint hash mismatch or missing pin")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) or not isinstance(v, torch.Tensor)
                                                     for k, v in state.items()):
        raise ValueError("Expected tensor-only state dictionary")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("State dictionary keys do not match the selected architecture")
    if any(state[k].shape != expected[k].shape or not torch.isfinite(state[k]).all() for k in expected):
        raise ValueError("Invalid tensor shape or non-finite weights")
    model.load_state_dict(state, strict=True)
    model.eval()


class S600CrisisAdapter(LazyStaticCrisisAdapter):
    """ST01c architecture/tokenizer + explicitly pinned FINAL research weights.

    Existing calibration is reused only as an unvalidated experimental decoding
    policy. It is not newly calibrated and cannot authorize production use.
    """
    version = "crisis-s600-final-uncalibrated/2026-10-18"

    def __init__(self, *, weights, weights_sha256, **kwargs):
        super().__init__(**kwargs)
        self.weights = Path(weights).resolve()
        self.weights_sha256 = weights_sha256
        self._overlay_lock = threading.Lock()
        self._overlay_loaded = False

    def _load(self):
        classifier = super()._load()
        with self._overlay_lock:
            if not self._overlay_loaded:
                load_weights_only(classifier.model, self.weights, self.weights_sha256)
                self._overlay_loaded = True
        return classifier

    def assess(self, latest_user_message):
        prediction = dict(self._load().predict(latest_user_message))
        prediction.update(model_id="S600_full_encoder_final_update440_seed42",
            weights_sha256=self.weights_sha256, calibration_status="inherited_ST01c_NOT_validated_for_S600",
            deployment_authorized=False)
        return bridge_static_crisis(prediction)


class FinalBoundaryAdapter:
    version = "boundary-conservative-final/2026-10-18"

    def __init__(self, root, device="cpu"):
        self.root, self.device, self._guard = Path(root).resolve(), device, None

    def _load(self):
        import sys
        if self._guard is None:
            sys.path.insert(0, str(self.root))
            import boundary
            Path(boundary.__file__).resolve().relative_to(self.root)
            from boundary.conservative import ConservativeGuard
            from boundary.recalibrated import RecalibratedBackend
            self._guard = ConservativeGuard(RecalibratedBackend(self.root / "model",
                self.root / "calibration/candidate.json", self.device, 2))
        return self._guard

    def predict(self, text):
        return self._load().predict(text)

    def readiness(self):
        return {"ready": (self.root / "model/model.safetensors").is_file(), "clinical_deployment_ready": False}
