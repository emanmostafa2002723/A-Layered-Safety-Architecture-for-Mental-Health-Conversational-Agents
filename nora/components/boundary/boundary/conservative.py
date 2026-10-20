"""Current checkpoint plus auditable clarification/reason repairs, not retraining.

The targeted reason repair can never change restrict into allow: it requires
an existing clinical-treatment reason and removes only an unsupported medicine
reason contributed by the bare 'prescribe' rule for a non-drug clinical task.
"""
import argparse
from contextlib import redirect_stdout
import json
import re
import sys

from .corrected import CorrectiveGuard
from .mvp import worker
from .recalibrated import RecalibratedBackend
from .schema import Decision, normalize
from .policy import DIRECT
from .integration import resolve


NONDRUG_TASK = re.compile(r"\s+(?:(?:the\s+next|my|a|an|the)\s+)?(?:individualized\s+|individualised\s+)?clinical\s+(?:homework|psychotherapy|therapy|exposure)(?:\b)", re.I)
MEDICINE = re.compile(r"\b(?:medicines?|medications?|drugs?|pills?|tablets?|dos(?:e|age)s?|mg|milligrams?|prescriptions?|antidepressants?|sertraline|fluoxetine|lithium|escitalopram|venlafaxine|buspirone|methylphenidate|trazodone|aripiprazole|duloxetine|quetiapine|atomoxetine)\b", re.I)


class ConservativeGuard(CorrectiveGuard):
    def predict(self, text):
        result = super().predict(text)
        if result["decision_kind"] != "restrict" or not {"clinical_treatment", "medication_decision"}.issubset(result["reasons"]): return result
        evidence = result.get("evidence", {}); raw = evidence.get("classifier", {})
        if "medication_decision" in raw.get("reasons", []) or set(evidence.get("rule_ids", [])) & {"request_2", "request_3"}: return result
        normalized = normalize(text)
        if MEDICINE.search(normalized): return result
        matches = list(DIRECT[4][1].finditer(normalized))
        if not matches or any(match.group().lower() != "prescribe" or not NONDRUG_TASK.match(normalized, match.end()) for match in matches): return result
        reasons = [reason for reason in result["reasons"] if reason != "medication_decision"]
        evidence = {**evidence, "reason_repairs": ["bare_prescribe_non_drug_clinical_task_not_medication"],
                    "removed_rule_reason": "medication_decision", "restriction_retained": True}
        return Decision("restrict", reasons, result["safety_signals"], result["status"], result["complete_input_seen"],
            result["backend"], evidence, result["latency_ms"]).to_dict()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--calibration", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu"); parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--text", action="append"); parser.add_argument("--raw", action="store_true"); parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    with redirect_stdout(sys.stderr): backend = RecalibratedBackend(args.checkpoint, args.calibration, args.device, args.threads)
    guard = ConservativeGuard(backend)
    with redirect_stdout(sys.stderr): warmup = guard.predict("What is CBT?")
    if warmup["status"] != "ok": raise RuntimeError("Inference warm-up failed")
    if args.worker: return worker(guard)
    if not args.text: parser.error("Supply --text or --worker")
    values = []
    for text in args.text:
        if args.raw: values.append(backend.raw(text))
        else:
            decision = guard.predict(text)
            values.append({"decision": decision, "policy_preview_without_crisis_model": resolve({"tier": "normal"}, decision)})
    print(json.dumps({"results": values, "profile": "current_weights_with_clarification_and_reason_repairs", "clinical_deployment_ready": False}, indent=2))
    return 0


if __name__ == "__main__": raise SystemExit(main())
