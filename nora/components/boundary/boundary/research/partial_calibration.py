"""Nullable-label calibration components; no training or deployment changes.

Inputs must be the dedicated calibration partition of a later frozen dataset.
Results on the fitting partition are not independent validation or clinical
accuracy. This module is not connected to the legacy baseline entrypoint.
"""
from __future__ import annotations

import math

import numpy as np

from ..schema import LABELS
from .contracts import targets_and_mask
from .partial_data import partial_metrics


def calibrated_probabilities(logits, temperatures):
    logits = np.asarray(logits, dtype=np.float64)
    temperatures = np.asarray(temperatures, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != len(LABELS) or not np.isfinite(logits).all():
        raise ValueError("Expected a finite N by four logit matrix")
    if temperatures.shape != (len(LABELS),) or not np.isfinite(temperatures).all() or (temperatures <= 0).any():
        raise ValueError("Expected four positive finite temperatures")
    # Strictly interior float64 probabilities make threshold 0/1 explicit
    # all-positive/no-positive endpoints under the same >= decision rule.
    values = logits / temperatures
    probabilities = np.exp(-np.logaddexp(0, -values))
    return np.clip(probabilities, np.nextafter(0., 1.), np.nextafter(1., 0.))


def observed_arrays(rows):
    if not rows:
        raise ValueError("Calibration partition is empty")
    arrays = [targets_and_mask(row["targets"]) for row in rows]
    return (np.asarray([values for values, _ in arrays], dtype=np.float64),
            np.asarray([mask for _, mask in arrays], dtype=bool))


def reliability(rows, probabilities, bins=10):
    truth, mask = observed_arrays(rows)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != truth.shape or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid probability matrix")
    if type(bins) is not int or bins < 1:
        raise ValueError("Positive integer bin count required")
    result = {}
    for index, label in enumerate(LABELS):
        values, actual = probabilities[mask[:, index], index], truth[mask[:, index], index]
        if not len(actual):
            result[label] = {"observed": 0, "brier": None, "ece": None, "bins": []}
            continue
        memberships = np.minimum((values * bins).astype(int), bins - 1)
        table = []
        error = 0.
        for number in range(bins):
            selected = memberships == number
            if not selected.any():
                continue
            mean_probability, positive_rate = float(values[selected].mean()), float(actual[selected].mean())
            count = int(selected.sum())
            error += count / len(actual) * abs(mean_probability - positive_rate)
            table.append({"bin": number, "count": count, "mean_probability": mean_probability, "positive_rate": positive_rate})
        result[label] = {"observed": len(actual), "brier": float(np.mean((values - actual) ** 2)),
                         "ece": float(error), "bins": table}
    return result


def threshold_candidates(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or ((values <= 0) | (values >= 1)).any():
        raise ValueError("Candidate scores must be finite and strictly inside zero and one")
    return np.unique(np.concatenate(([0.], values, [1.])))


def fit(rows, logits, *, partition, minimum_recall=.95, maximum_benign_diversion=.02,
        minimum_positive_support=200, minimum_benign_support=500):
    if partition != "calibration":
        raise ValueError("Calibration cannot fit on train, selection or test")
    if not 0 < minimum_recall <= 1 or not 0 <= maximum_benign_diversion <= 1:
        raise ValueError("Invalid predeclared recall/diversion constraints")
    if type(minimum_positive_support) is not int or minimum_positive_support < 1 or type(minimum_benign_support) is not int or minimum_benign_support < 1:
        raise ValueError("Positive integer support requirements required")
    truth, mask = observed_arrays(rows)
    logits = np.asarray(logits, dtype=np.float64)
    if logits.shape != truth.shape or not np.isfinite(logits).all():
        raise ValueError("Logit/record shape or values invalid")
    from scipy.optimize import minimize_scalar
    temperatures, support = [], {}
    for index, label in enumerate(LABELS):
        known = mask[:, index]
        actual, values = truth[known, index], logits[known, index]
        positives = int(actual.sum())
        negatives = len(actual) - positives
        if not positives or not negatives:
            raise ValueError(f"{label}: observed positive and negative calibration targets required")
        fitted = minimize_scalar(lambda log_temp: float(np.mean(
            np.logaddexp(0, values / np.exp(log_temp)) - actual * values / np.exp(log_temp))),
            method="bounded", bounds=(-2., 2.))
        if not fitted.success or not np.isfinite(fitted.fun):
            raise ValueError("Temperature optimization failed; no silent default")
        temperatures.append(float(np.exp(fitted.x)))
        support[label] = {"observed_positive": positives, "observed_negative": negatives}
    probabilities = calibrated_probabilities(logits, temperatures)
    thresholds, candidate_counts = [], {}
    for index, label in enumerate(LABELS):
        actual = truth[mask[:, index], index].astype(bool)
        scores = probabilities[mask[:, index], index]
        # Highest threshold satisfying each recall constraint minimizes every
        # class's positive set, hence also their union on fully known benign
        # rows. If that union fails diversion, no lower threshold can fix it.
        positives = np.sort(scores[actual])[::-1]
        required = math.ceil(minimum_recall * len(positives))
        thresholds.append(float(positives[required - 1]))
        candidate_counts[label] = len(threshold_candidates(scores))
    metrics = partial_metrics(rows, probabilities, thresholds)
    benign_support = metrics["known_allowed_support"]
    benign_diversion = metrics["known_allowed_flagged"] / benign_support if benign_support else None
    failures = []
    for label in LABELS:
        if support[label]["observed_positive"] < minimum_positive_support:
            failures.append(f"{label}: insufficient observed positive support")
        if metrics["classes"][label]["recall"] < minimum_recall:
            failures.append(f"{label}: recall constraint not met")
    if benign_support < minimum_benign_support:
        failures.append("Insufficient fully observed benign support")
    if benign_diversion is None or benign_diversion > maximum_benign_diversion:
        failures.append("Joint benign diversion constraint not met; no feasible lower thresholds")
    return {"temperatures": temperatures, "thresholds": thresholds, "decision_comparator": ">=",
            "probability_transform": "strictly_interior_float64_sigmoid_after_temperature",
            "threshold_search": "observed score boundary attaining maximum threshold under each recall constraint; zero/one endpoints included",
            "candidate_counts_including_endpoints": candidate_counts,
            "fitted_on": "calibration_only", "support": support,
            "constraints": {"minimum_recall": minimum_recall, "maximum_benign_diversion": maximum_benign_diversion,
                            "minimum_positive_support": minimum_positive_support, "minimum_benign_support": minimum_benign_support},
            "joint_benign_diversion_fit": benign_diversion,
            "fit_constraints_met": not failures, "failure_reasons": failures,
            "fit_metrics_not_test": metrics,
            "uncalibrated_reliability_fit": reliability(rows, calibrated_probabilities(logits, [1.] * len(LABELS))),
            "calibrated_reliability_fit": reliability(rows, probabilities),
            "deployment_ready": False, "clinical_accuracy": None,
            "warning": "Fitting results are optimistic; require frozen independent evaluation and family-level uncertainty before promotion."}
