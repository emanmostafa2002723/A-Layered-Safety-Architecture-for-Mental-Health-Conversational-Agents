"""Candidate confidence recalibration with explicit legacy-equivalent routing.

No newly selected routing threshold is silently installed. The original weights,
model/calibration artifacts and entrypoint remain intact. Confidence validation
is on authored weak labels, never advertised as clinical confidence.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

from .mvp import MVPBackend, MVPGuard, worker
from .schema import LABELS, normalize
from .sources import sha256
from .integration import resolve
from .research.calibration_v2 import transformed


class RecalibratedBackend(MVPBackend):
    name = "fine_tuned_deberta_recalibrated_research_candidate"

    def __init__(self, checkpoint, calibration, device="cpu", threads=2):
        super().__init__(checkpoint, device, threads)
        artifact = json.loads(Path(calibration).read_text(encoding="utf-8"))
        if artifact.get("schema") != "boundary-confidence-candidate-2.0" or artifact.get("labels") != list(LABELS):
            raise ValueError("Unknown confidence artifact")
        if artifact.get("model_sha256") != self.manifest["model_sha256"] or artifact.get("legacy_calibration_sha256") != sha256(Path(checkpoint) / "calibration.json"):
            raise ValueError("Candidate belongs to different model/calibration artifacts")
        if artifact.get("decision_mode") != "legacy_equivalent_logit_comparison" or artifact.get("clinical_deployment_ready") is not False:
            raise ValueError("Unsupported decision mode or clinical claim")
        self.confidence = artifact["confidence_transform"]
        transformed(np.zeros((1, 4)), self.confidence)
        self.logit_cutoffs = np.asarray(artifact["legacy_equivalent_logit_cutoffs"], dtype=float)
        if self.logit_cutoffs.shape != (4,) or not np.isfinite(self.logit_cutoffs).all():
            raise ValueError("Finite decision cutoffs required")
        self.candidate_sha256 = sha256(calibration)

    def raw(self, text):
        encoded = self.tokenizer(normalize(text), truncation=False, return_tensors="pt")
        if encoded["input_ids"].shape[1] > self.max_length:
            raise OverflowError("Complete input exceeds trained model window; not truncated")
        with self.torch.inference_mode():
            logits = self.model(**{key: value.to(self.device) for key, value in encoded.items()}).logits.float().cpu().numpy()
        probabilities = expit(transformed(logits, self.confidence))[0]
        display_thresholds = expit(transformed(self.logit_cutoffs[None, :], self.confidence))[0]
        return {"reasons": [label for index, label in enumerate(LABELS) if logits[0, index] >= self.logit_cutoffs[index]],
                "probabilities": dict(zip(LABELS, map(float, probabilities))),
                "thresholds": dict(zip(LABELS, map(float, display_thresholds))),
                "raw_logits": dict(zip(LABELS, map(float, logits[0]))),
                "raw_logit_thresholds": dict(zip(LABELS, map(float, self.logit_cutoffs))),
                "decision_space": "raw_logits_legacy_routing_equivalent",
                "confidence_method": self.confidence["kind"], "calibration_candidate_sha256": self.candidate_sha256,
                "classifier_only": True, "clinical_probabilities": False,
                "calibration_validation_scope": "AI_authored_weak_policy_examples_not_clinical_population"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--calibration", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu"); parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--text", action="append"); parser.add_argument("--raw", action="store_true"); parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr): backend = RecalibratedBackend(args.checkpoint, args.calibration, args.device, args.threads)
    guard = MVPGuard(backend)
    if args.worker:
        with redirect_stdout(sys.stderr): warmup = guard.predict("What is cognitive behavioral therapy?")
        if warmup["status"] != "ok": raise RuntimeError("Inference warm-up failed")
        return worker(guard)
    if not args.text: parser.error("Supply --text or --worker")
    results = []
    for text in args.text:
        if args.raw: results.append(backend.raw(text))
        else:
            decision = guard.predict(text)
            results.append({"decision": decision, "policy_preview_without_crisis_model": resolve({"tier": "normal"}, decision)})
    print(json.dumps({"results": results, "research_candidate": True, "clinical_deployment_ready": False}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
