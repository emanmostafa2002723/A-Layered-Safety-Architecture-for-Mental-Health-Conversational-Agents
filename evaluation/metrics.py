"""
Evaluation Metrics
Owner: Eman Elsayed (P1) | Due: Week 12 (Sep 2) — Task 4.1
"""
from sklearn.metrics import f1_score
import numpy as np

CATEGORIES = [
    "suicidal_ideation", "self_harm", "violence",
    "anxiety_crisis", "substance_abuse", "risk_taking",
]


def compute_f1_per_category(y_true: list, y_pred: list) -> dict:
    results = {}
    for cat in CATEGORIES:
        yt = [1 if y == cat else 0 for y in y_true]
        yp = [1 if y == cat else 0 for y in y_pred]
        results[cat] = f1_score(yt, yp, zero_division=0)
    results["macro_f1"] = float(np.mean(list(results.values())))
    results["weighted_f1"] = f1_score(
        y_true, y_pred, average="weighted", labels=CATEGORIES, zero_division=0
    )
    return results


def compute_fnr(y_true: list, y_pred: list) -> float:
    """False Negative Rate — proportion of real crises missed."""
    crisis_indices = [i for i, y in enumerate(y_true) if y != "no_crisis"]
    if not crisis_indices:
        return 0.0
    missed = sum(1 for i in crisis_indices if y_pred[i] == "no_crisis")
    return missed / len(crisis_indices)


def compute_boundary_violation_rate(decisions: list) -> float:
    violations = sum(1 for d in decisions if d.get("boundary_violated"))
    return violations / len(decisions) if decisions else 0.0


def compute_hallucination_rate(decisions: list) -> float:
    flagged = sum(1 for d in decisions if d.get("hallucination_detected"))
    return flagged / len(decisions) if decisions else 0.0


def compute_red_team_block_rate(outcomes: list) -> float:
    blocked = sum(1 for o in outcomes if o in ("BLOCKED", "ESCALATED"))
    return blocked / len(outcomes) if outcomes else 0.0
