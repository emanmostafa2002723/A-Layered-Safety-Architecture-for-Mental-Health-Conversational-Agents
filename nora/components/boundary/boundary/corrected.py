"""Opt-in corrected-model inference with narrow, auditable clarification checks.

No automatic default replacement, no NLI, no diagnosis of a person. Broad
confidence-based rejection is not implemented without adequate validation.
"""
import argparse
from contextlib import redirect_stdout
import json
import re
import sys
import time

from .recalibrated import RecalibratedBackend
from .mvp import MVPGuard, worker
from .schema import Decision, normalize
from .policy import inspect
from .integration import resolve


REFERENT = re.compile(
    r"^\s*(?:(?:should|can|could|may)\s+i\s+(?:keep\s+taking|go\s+off|cut\s+down\s+on|take\s+another\s+of)\s+(?:it|them|that|this)"
    r"|(?:would|could)\s+(?:taking\s+more|taking\s+less|stopping\s+it)\s+(?:help|be\s+safer)"
    r"|(?:is\s+it\s+(?:safe|okay)|would\s+it\s+be\s+(?:safe|okay))\s+to\s+(?:mix|combine)\s+(?:them|those|these))\s*(?:tonight|now|today)?\s*[?.!]*\s*$", re.I)


class CorrectiveGuard(MVPGuard):
    def predict(self, text):
        start = time.perf_counter()
        try:
            normalized = normalize(text)
        except (ValueError, TypeError): return super().predict(text)
        if not REFERENT.fullmatch(normalized): return super().predict(text)
        safety = inspect(normalized)["safety_signals"]
        return Decision("uncertain", safety_signals=safety, backend=self.backend.name,
            evidence={"rule_ids": ["unresolved_action_referent_requires_clarification"], "classifier_bypassed": True,
                      "clinical_inference_made": False}, latency_ms=round(1000 * (time.perf_counter()-start), 3)).to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--calibration", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu"); parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--text", action="append"); parser.add_argument("--raw", action="store_true"); parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr): backend = RecalibratedBackend(args.checkpoint, args.calibration, args.device, args.threads)
    guard = CorrectiveGuard(backend)
    with redirect_stdout(sys.stderr): warmup = guard.predict("What is CBT?")
    if warmup["status"] != "ok": raise RuntimeError("Candidate inference warm-up failed")
    if args.worker: return worker(guard)
    if not args.text: parser.error("Supply --text or --worker")
    results = []
    for text in args.text:
        if args.raw: results.append(backend.raw(text))
        else:
            decision = guard.predict(text)
            results.append({"decision": decision, "policy_preview_without_crisis_model": resolve({"tier": "normal"}, decision)})
    print(json.dumps({"results": results, "corrective_research_candidate": True, "clinical_deployment_ready": False}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
