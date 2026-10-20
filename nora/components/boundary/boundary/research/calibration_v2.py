"""Regularized post-hoc calibration; probability quality is not routing quality.

No training, dataset I/O or clinical claims. Per-family weights avoid treating
contrasts as independent evidence. All transforms are monotone within a label.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from ..schema import LABELS
from .partial_calibration import calibrated_probabilities


def arrays(rows, logits):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray([[row["targets"][label] for label in LABELS] for row in rows], dtype=float)
    if z.shape != y.shape or z.ndim != 2 or z.shape[1] != 4 or not len(rows) or not np.isfinite(z).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Finite, fully labelled nonempty four-column data required")
    counts = {family: sum(row["family"] == family for row in rows) for family in {r["family"] for r in rows}}
    weights = np.asarray([1 / counts[row["family"]] for row in rows])
    weights /= weights.sum()
    return y, z, weights


def transformed(logits, artifact):
    z = np.asarray(logits, dtype=float); slope = np.asarray(artifact["slopes"], dtype=float); bias = np.asarray(artifact["biases"], dtype=float)
    if z.ndim != 2 or z.shape[1] != 4 or slope.shape != (4,) or bias.shape != (4,) or not np.isfinite(z).all() or not np.isfinite(slope).all() or not np.isfinite(bias).all() or (slope <= 0).any():
        raise ValueError("Finite monotone four-label transform required")
    return z * slope + bias


def nll(y, z, weights):
    loss = np.where(y == 1, np.logaddexp(0, -z), np.logaddexp(0, z))
    return float(np.sum(weights[:, None] * loss) / 4)


def fit(rows, logits, kind, regularization=0.1):
    y, z, weights = arrays(rows, logits)
    if kind not in {"identity", "temperature", "sigmoid"} or not np.isfinite(regularization) or regularization < 0:
        raise ValueError("Unknown method or invalid regularization")
    if ((y.sum(0) == 0) | (y.sum(0) == len(y))).any(): raise ValueError("Every label needs both outcomes")
    if kind == "identity":
        return {"kind": kind, "slopes": [1.] * 4, "biases": [0.] * 4, "optimizer_ok": True,
                "at_parameter_bound": False, "regularization": 0., "fit_families": len({r["family"] for r in rows})}
    # Regularization is toward identity, not toward high confidence. Parameter
    # bounds guard optimization; hitting one disqualifies a fitted candidate.
    bounds = [(-math.log(20), math.log(20))] if kind == "temperature" else [(-math.log(20), math.log(20))] * 4 + [(-10., 10.)] * 4
    def parameters(theta):
        return (np.repeat(np.exp(-theta[0]), 4), np.zeros(4)) if kind == "temperature" else (np.exp(theta[:4]), theta[4:])
    def objective(theta):
        a, b = parameters(theta)
        return nll(y, z * a + b, weights) + regularization * float(np.mean(np.asarray(theta) ** 2))
    optimum = minimize(objective, np.zeros(len(bounds)), method="L-BFGS-B", bounds=bounds)
    if not optimum.success or not np.isfinite(optimum.fun) or not np.isfinite(optimum.x).all():
        raise ValueError("Calibration optimization failed; no silent fallback")
    a, b = parameters(optimum.x)
    return {"kind": kind, "slopes": list(map(float, a)), "biases": list(map(float, b)),
            "regularization": regularization, "optimizer_ok": True,
            "at_parameter_bound": any(min(abs(value - lo), abs(value - hi)) < 1e-4 for value, (lo, hi) in zip(optimum.x, bounds)),
            "fit_families": len({r["family"] for r in rows})}


def metrics(rows, logits, artifact):
    y, z, weights = arrays(rows, logits); scaled = transformed(z, artifact); p = expit(scaled)
    classes = {}
    for index, label in enumerate(LABELS):
        bins = []; ece = 0.
        memberships = np.minimum((p[:, index] * 10).astype(int), 9)
        for number in range(10):
            selected = memberships == number
            if not selected.any(): continue
            mass = float(weights[selected].sum())
            predicted = float(np.average(p[selected, index], weights=weights[selected]))
            actual = float(np.average(y[selected, index], weights=weights[selected]))
            ece += mass * abs(predicted - actual)
            bins.append({"bin": number, "rows": int(selected.sum()), "family_weight_mass": mass,
                         "mean_probability": predicted, "positive_rate": actual})
        classes[label] = {"positive_rows": int(y[:, index].sum()),
            "positive_families": len({row["family"] for row in rows if row["targets"][label]}),
            "brier": float(np.sum(weights * (p[:, index] - y[:, index]) ** 2)), "ece_10_bins": ece, "bins": bins}
    return {"records": len(rows), "families": len({r["family"] for r in rows}),
            "nll": nll(y, scaled, weights), "brier": float(np.sum(weights[:, None] * (p - y) ** 2) / 4),
            "endpoint_probabilities": int(((p == 0) | (p == 1)).sum()),
            "near_endpoint_probabilities_1e_minus12": int(((p <= 1e-12) | (p >= 1 - 1e-12)).sum()),
            "classes": classes, "clinical_confidence_validated": False,
            "weighting": "equal_family_mass_then_equal_rows_within_family", "prevalence": "authored_not_deployment"}


def cv_regularization(rows, logits, kind, penalties=(.01, .1, 1.)):
    z = np.asarray(logits, dtype=float)
    families = sorted({row["family"] for row in rows})
    # Distribute each authored slice across folds without splitting a family.
    fold_for = {}
    for slice_name in sorted({row["slice"] for row in rows}):
        subset = sorted({row["family"] for row in rows if row["slice"] == slice_name})
        for index, family in enumerate(subset): fold_for[family] = index % 3
    outcomes = []
    for penalty in penalties:
        scores = []; failures = []
        for fold in range(3):
            train = [i for i, row in enumerate(rows) if fold_for[row["family"]] != fold]
            valid = [i for i, row in enumerate(rows) if fold_for[row["family"]] == fold]
            if not train or not valid: failures.append("empty group fold"); continue
            try:
                artifact = fit([rows[i] for i in train], z[train], kind, penalty)
                if artifact["at_parameter_bound"]: raise ValueError("parameter-bound fit")
                scores.append(metrics([rows[i] for i in valid], z[valid], artifact)["nll"])
            except ValueError as error: failures.append(str(error))
        outcomes.append({"regularization": penalty, "mean_cv_nll": float(np.mean(scores)) if len(scores) == 3 else None,
                         "fold_failures": failures})
    valid = [item for item in outcomes if item["mean_cv_nll"] is not None]
    if not valid: raise ValueError("All group-cross-validation candidates failed")
    # Prefer stronger regularization in a numerical tie.
    best = min(valid, key=lambda item: (item["mean_cv_nll"], -item["regularization"]))
    artifact = fit(rows, z, kind, best["regularization"])
    return artifact, outcomes


def choose_method(candidates, rows, logits):
    scores = {name: metrics(rows, logits, value) for name, value in candidates.items()}
    identity = scores["identity"]
    eligible = [name for name, value in candidates.items() if not value["at_parameter_bound"]
                and scores[name]["nll"] <= identity["nll"] + 1e-12
                and scores[name]["brier"] <= identity["brier"] + 1e-12
                and scores[name]["endpoint_probabilities"] == 0]
    if not eligible: return "identity", scores, False
    chosen = min(eligible, key=lambda name: (scores[name]["nll"], 0 if name == "identity" else 1))
    return chosen, scores, chosen != "identity"


def threshold_policy(rows, logits, minimum_recall=.95, minimum_precision=.8, maximum_benign_diversion=.05,
                     minimum_positive_rows=20, minimum_positive_families=10):
    y, z, _ = arrays(rows, logits); thresholds = []; failures = []
    for index, label in enumerate(LABELS):
        positive = y[:, index] == 1; count = int(positive.sum())
        families = {row["family"] for row in rows if row["targets"][label]}
        if count < minimum_positive_rows or len(families) < minimum_positive_families:
            failures.append(f"{label}: insufficient independent support for threshold promotion")
        candidates = np.unique(z[:, index]); feasible = []
        for threshold in candidates:
            pred = z[:, index] >= threshold
            tp = int((pred & positive).sum()); fp = int((pred & ~positive).sum())
            if count and tp / count >= minimum_recall and tp / (tp + fp) >= minimum_precision:
                feasible.append(float(threshold))
        thresholds.append(max(feasible) if feasible else None)
        if not feasible: failures.append(f"{label}: recall and precision jointly infeasible on these rows")
    if all(value is not None for value in thresholds):
        pred = z >= np.asarray(thresholds); benign = ~y.astype(bool).any(1)
        diversion = float(pred[benign].any(1).mean()) if benign.any() else None
        if diversion is None or diversion > maximum_benign_diversion:
            failures.append("Joint benign diversion exceeds predeclared constraint")
    else: diversion = None
    return {"raw_logit_thresholds": thresholds, "eligible": not failures, "failures": failures,
            "joint_benign_diversion": diversion,
            "constraints": {"minimum_recall": minimum_recall, "minimum_precision": minimum_precision,
                            "maximum_benign_diversion": maximum_benign_diversion,
                            "minimum_positive_rows": minimum_positive_rows, "minimum_positive_families": minimum_positive_families}}


def legacy_logit_cutoffs(temperatures, thresholds):
    """Invert the ACTUAL old floating-point predicate, not a rounded logit(p).

    This preserves legacy routing while probability display is recalibrated.
    It does not repair classifier discrimination or establish new safety limits.
    """
    temperatures = np.asarray(temperatures, dtype=float); thresholds = np.asarray(thresholds, dtype=float)
    if temperatures.shape != (4,) or thresholds.shape != (4,) or not np.isfinite(temperatures).all() or not np.isfinite(thresholds).all() or (temperatures <= 0).any() or ((thresholds <= 0) | (thresholds >= 1)).any():
        raise ValueError("Interior finite legacy thresholds required")
    low = -1024 * temperatures; high = 1024 * temperatures
    for _ in range(100):
        midpoint = (low + high) / 2
        accepted = calibrated_probabilities(midpoint[None, :], temperatures)[0] >= thresholds
        high = np.where(accepted, midpoint, high); low = np.where(accepted, low, midpoint)
    return high


def group_bootstrap_probability_delta(rows, logits, before, after, repetitions=500):
    # Paired family bootstrap: do not treat contrasts/substitutions as independent.
    z = np.asarray(logits); groups = sorted({row["family"] for row in rows})
    y = np.asarray([[row["targets"][label] for label in LABELS] for row in rows])
    zb, za = transformed(z, before), transformed(z, after)
    losses = []
    for family in groups:
        mask = np.asarray([row["family"] == family for row in rows])
        a = np.where(y[mask] == 1, np.logaddexp(0, -za[mask]), np.logaddexp(0, za[mask])).mean()
        b = np.where(y[mask] == 1, np.logaddexp(0, -zb[mask]), np.logaddexp(0, zb[mask])).mean()
        losses.append(a - b)
    random = np.random.default_rng(20261018)
    means = np.asarray(losses)[random.integers(0, len(groups), (repetitions, len(groups)))].mean(1)
    return {"method": "paired_authored_family_bootstrap", "repetitions": repetitions,
            "nll_delta_after_minus_before": float(np.mean(losses)),
            "percentile_95": list(map(float, np.percentile(means, [2.5, 97.5]))),
            "clinical_population_interval": False, "warning": "Authored family dependence and designed sampling limit external inference"}
