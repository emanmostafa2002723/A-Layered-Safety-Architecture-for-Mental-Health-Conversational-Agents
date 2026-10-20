"""Research-MVP inference/JSONL worker using actual fine-tuned DeBERTa weights.

Raw mode reports the classifier alone. Guarded mode additionally applies
explicit transformation/context checks and the separate safety screen.
Neither mode diagnoses people or generates a chatbot response.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

from .schema import LABELS, Decision, normalize
from .policy import inspect
from .integration import resolve
from .sources import sha256
from .research.partial_calibration import calibrated_probabilities


TRANSFORM = re.compile(
    r'''^\s*(?:please\s+)?(?:translate|proofread|rewrite|edit|correct)\b[^"'\n]{0,180}'''
    r'''(?:without answering(?: it)?|do not answer(?: it)?|spelling only|grammar only)\s*:?\s*'''
    r'''(?P<quote>["'])(?P<quoted>.*?)(?P=quote)(?P<tail>.*)$''', re.I | re.S)
MISSING = re.compile(
    r"^\s*(?:(?:should|can|could|may|would)\s+(?:i|we|he|she|they)\s+"
    r"(?:stop|continue|discontinue|increase|reduce|double|switch|take(?:\s+(?:more|less))?|combine)"
    r"\s+(?:of\s+)?(?:it|that|this|them)(?:\s+(?:now|tonight|today|tomorrow|again|instead|as well))*"
    r"|would\s+(?:increasing|reducing|stopping|continuing|discontinuing|doubling|taking)\s+(?:it|that|this|them)\s+be\s+(?:better|safe|okay)\s+for\s+me"
    r"|is\s+it\s+(?:okay|safe)\s+for\s+me\s+to\s+(?:stop|continue|discontinue|increase|reduce|double|take)\s+(?:it|that|this|them)(?:\s+now)?)\s*[?.!]*\s*$", re.I)


def actionable_text(text):
    match = TRANSFORM.fullmatch(text)
    if not match:
        return text, False
    # Transformation applies only to the quoted span. An additional request
    # after the closing quote still reaches the model and safety screen.
    tail = match["tail"].strip(" .;\n")
    return tail, True


class MVPBackend:
    name = "fine_tuned_deberta_research_mvp"

    def __init__(self, checkpoint, device="cpu", threads=2):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        checkpoint = Path(checkpoint).resolve()
        self.manifest = json.loads((checkpoint / "training_manifest.json").read_text(encoding="utf-8"))
        self.calibration = json.loads((checkpoint / "calibration.json").read_text(encoding="utf-8"))
        if self.manifest.get("labels") != list(LABELS) or self.manifest.get("encoder_fine_tuned") is not True or not self.manifest.get("updates"):
            raise ValueError("A completed actual four-label training checkpoint is required")
        if sha256(checkpoint / "model.safetensors") != self.manifest["model_sha256"]:
            raise ValueError("Checkpoint weight identity changed")
        self.temperatures = np.asarray(self.calibration["temperatures"], dtype=float)
        self.thresholds = np.asarray(self.calibration["thresholds"], dtype=float)
        if self.thresholds.shape != (4,) or not np.isfinite(self.thresholds).all() or ((self.thresholds < 0) | (self.thresholds > 1)).any():
            raise ValueError("Invalid threshold artifact")
        calibrated_probabilities(np.zeros((1, 4)), self.temperatures)
        torch.set_num_threads(threads)
        self.torch, self.device = torch, torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, fix_mistral_regex=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True).float().to(self.device).eval()
        if self.model.config.num_labels != 4 or [self.model.config.id2label[i] for i in range(4)] != list(LABELS):
            raise ValueError("Wrong trained output head/label order")
        self.max_length = self.manifest["max_length"]

    def raw(self, text):
        text = normalize(text)
        encoded = self.tokenizer(text, truncation=False, return_tensors="pt")
        if encoded["input_ids"].shape[1] > self.max_length:
            raise OverflowError("Complete input exceeds the trained model window; not truncated")
        with self.torch.inference_mode():
            logits = self.model(**{key: value.to(self.device) for key, value in encoded.items()}).logits.float().cpu().numpy()
        probabilities = calibrated_probabilities(logits, self.temperatures)[0]
        return {"reasons": [label for i, label in enumerate(LABELS) if probabilities[i] >= self.thresholds[i]],
                "probabilities": dict(zip(LABELS, map(float, probabilities))),
                "thresholds": dict(zip(LABELS, map(float, self.thresholds))),
                "classifier_only": True, "clinical_probabilities": False}


class MVPGuard:
    def __init__(self, backend):
        self.backend = backend

    def predict(self, text):
        start = time.perf_counter()
        trace = {"reasons": [], "safety_signals": [], "rule_ids": []}
        try:
            text = normalize(text)
            action, transformed = actionable_text(text)
            trace = inspect(action) if action else {"reasons": [], "safety_signals": [], "rule_ids": []}
            if len(text.encode("utf-8")) > 16384:
                raise OverflowError("Input byte limit")
            if MISSING.fullmatch(text):
                result = Decision("uncertain", safety_signals=trace["safety_signals"], backend=self.backend.name,
                    evidence={"rule_ids": ["unresolved_referent_requires_context"], "classifier_bypassed": True})
            elif transformed and not action:
                result = Decision("allow", backend=self.backend.name,
                    evidence={"rule_ids": ["explicit_quoted_transformation_only"], "classifier_bypassed": True})
            else:
                raw = self.backend.raw(action)
                reasons = sorted(set(raw["reasons"]) | set(trace["reasons"]))
                result = Decision("restrict" if reasons else "allow", reasons, trace["safety_signals"], backend=self.backend.name,
                    evidence={"classifier": raw, "rule_ids": trace["rule_ids"], "quoted_transformation_removed": transformed,
                              "classifier_only": False, "calibration_fit_is_not_clinical_validation": True})
        except (TypeError, ValueError):
            result = Decision("unavailable", safety_signals=trace["safety_signals"], status="invalid_input_or_backend_output", complete_input_seen=False, backend=self.backend.name)
        except OverflowError:
            signal_text = text if isinstance(text, str) and text.strip() else "Input unavailable"
            result = Decision("unavailable", safety_signals=inspect(signal_text)["safety_signals"], status="input_too_long", complete_input_seen=False, backend=self.backend.name)
        except Exception:
            result = Decision("unavailable", safety_signals=trace["safety_signals"], status="backend_unavailable", backend=self.backend.name)
        result.latency_ms = round(1000 * (time.perf_counter() - start), 3)
        return result.to_dict()


def worker(guard):
    print(json.dumps({"schema_version": "worker-ready/1.0", "component": "boundary", "status": "ready"}), flush=True)
    while True:
        line = sys.stdin.buffer.readline(32769)
        if not line: return 0
        request_id = "unknown"
        try:
            if len(line) > 32768 or not line.endswith(b"\n"): raise ValueError("Input line too long")
            request = json.loads(line)
            request_id = request.get("request_id", "unknown")
            if request.get("schema_version") != "worker-request/1.0" or not isinstance(request_id, str) or not request_id:
                raise ValueError("Invalid transport envelope")
            if request.get("operation") == "shutdown": return 0
            if request.get("operation") != "predict" or not isinstance(request.get("payload"), dict) or set(request["payload"]) != {"latest_user_message"}:
                raise ValueError("Invalid operation/payload")
            with redirect_stdout(sys.stderr): result = guard.predict(request["payload"]["latest_user_message"])
            response = {"schema_version": "worker-response/1.0", "request_id": request_id, "status": "ok", "result": result}
        except Exception:
            response = {"schema_version": "worker-response/1.0", "request_id": request_id, "status": "error", "error_code": "component_request_failed"}
        print(json.dumps(response), flush=True)
        if len(line) > 32768 or not line.endswith(b"\n"): return 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=2); parser.add_argument("--text", action="append")
    parser.add_argument("--raw", action="store_true"); parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr): backend = MVPBackend(args.checkpoint, args.device, args.threads)
    guard = MVPGuard(backend)
    if args.worker: return worker(guard)
    if not args.text: parser.error("Supply --text or --worker")
    results = []
    for text in args.text:
        if args.raw:
            results.append(backend.raw(text))
        else:
            decision = guard.predict(text)
            results.append({"decision": decision, "policy_preview_without_crisis_model": resolve({"tier": "normal"}, decision)})
    print(json.dumps({"results": results, "research_mvp": True, "clinical_deployment_ready": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
