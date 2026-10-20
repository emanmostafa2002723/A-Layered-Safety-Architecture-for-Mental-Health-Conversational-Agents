"""Nullable-target data/metric adapter for later supervised boundary training.

No training loop, checkpoint write, inference or data labeling is performed.
Old weak_v3 trainer is not modified or silently pointed at this schema.
"""
from __future__ import annotations

import numpy as np

from ..schema import LABELS
from .contracts import targets_and_mask, validate_record


def encode_records(rows, tokenizer, registry, *, partition, max_length=192):
    if not rows:
        raise ValueError("Cannot encode an empty partition")
    for row in rows:
        validate_record(row, registry, partition=partition)
    encoded = tokenizer([row["text"] for row in rows], truncation=False)
    if len(encoded["input_ids"]) != len(rows):
        raise ValueError("Tokenizer row-count mismatch")
    if any(len(value) > max_length for value in encoded["input_ids"]):
        raise ValueError("Input exceeds configured length; do not silently truncate")
    output = []
    for index, row in enumerate(rows):
        values, mask = targets_and_mask(row["targets"])
        weight = row.get("sample_weight", 1.0)
        if type(weight) not in (int, float) or not np.isfinite(weight) or weight <= 0:
            raise ValueError("Sample weight must be a positive finite number")
        output.append({**{key: value[index] for key, value in encoded.items()},
                       "labels": values, "label_mask": mask, "sample_weight": float(weight)})
    return output


def observed_positive_weights(rows, maximum=10.0):
    if maximum < 1 or not np.isfinite(maximum):
        raise ValueError("Invalid positive-weight bound")
    if not rows:
        raise ValueError("No training rows")
    positives = np.zeros(len(LABELS), dtype=np.int64)
    negatives = np.zeros(len(LABELS), dtype=np.int64)
    for row in rows:
        values, mask = targets_and_mask(row["targets"])
        values, mask = np.asarray(values), np.asarray(mask, dtype=bool)
        positives += (mask & (values == 1)).astype(np.int64)
        negatives += (mask & (values == 0)).astype(np.int64)
    if (positives == 0).any() or (negatives == 0).any():
        raise ValueError("Each class needs observed positive and negative training targets")
    return np.clip(negatives / positives, 1.0, maximum)


def partial_metrics(rows, probabilities, thresholds):
    probabilities = np.asarray(probabilities, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if probabilities.shape != (len(rows), len(LABELS)) or not len(rows):
        raise ValueError("Probability matrix shape mismatch or empty input")
    if thresholds.shape != (len(LABELS),) or not np.isfinite(thresholds).all() or ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError("Invalid thresholds")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Invalid probabilities")
    arrays = [targets_and_mask(row["targets"]) for row in rows]
    truth = np.asarray([value for value, _ in arrays], dtype=bool)
    observed = np.asarray([mask for _, mask in arrays], dtype=bool)
    predicted = probabilities >= thresholds
    classes = {}
    for index, label in enumerate(LABELS):
        known = observed[:, index]
        actual, pred = truth[known, index], predicted[known, index]
        tp = int((actual & pred).sum())
        fp = int((~actual & pred).sum())
        fn = int((actual & ~pred).sum())
        tn = int((~actual & ~pred).sum())
        classes[label] = {"observed": int(known.sum()), "unknown": int((~known).sum()),
                          "support": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                          "precision": tp / (tp + fp) if tp + fp else None,
                          "recall": tp / (tp + fn) if tp + fn else None,
                          "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    complete = observed.all(1)
    known_positive = (truth & observed).any(1)
    known_negative = complete & ~truth.any(1)
    any_prediction = predicted.any(1)
    return {"records": len(rows), "classes": classes,
            "fully_observed_records": int(complete.sum()),
            "exact_match_support": int(complete.sum()),
            "exact_matches": int(((predicted == truth).all(1) & complete).sum()),
            "known_restricted_support": int(known_positive.sum()),
            "known_restricted_missed": int((known_positive & ~any_prediction).sum()),
            "known_allowed_support": int(known_negative.sum()),
            "known_allowed_flagged": int((known_negative & any_prediction).sum()),
            "any_boundary_ground_truth_unknown": int((~known_positive & ~known_negative).sum()),
            "scope": "raw_classifier_known_targets_only; no rules, uncertainty policy or emergency routing",
            "clinical_accuracy": None}
