"""Fail-closed nullable targets and provenance/split checks for the redesign.

These utilities do not assign clinical truth, train models, or replace the
legacy trainer. Unknown targets must never be converted into negative labels.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from ..schema import LABELS, text_key


def targets_and_mask(targets):
    if not isinstance(targets, dict) or set(targets) != set(LABELS):
        raise ValueError("Explicitly provide all four targets; unknown is null")
    values, mask = [], []
    for label in LABELS:
        value = targets[label]
        if value is not None and (type(value) is not int or value not in (0, 1)):
            raise ValueError("Targets must be integer 0/1 or null, not strings/confidence")
        values.append(float(value) if value is not None else 0.0)
        mask.append(float(value is not None))
    return values, mask


def target_disposition(targets):
    targets_and_mask(targets)
    if any(value == 1 for value in targets.values()):
        return "restricted"
    if all(value == 0 for value in targets.values()):
        return "allowed"
    return "unknown"


def consensus_targets(votes):
    """Agreement filter only; does not turn teacher agreement into human gold."""
    if len(votes) != 2:
        raise ValueError("Exactly two separately identified teacher votes required")
    families = set()
    for vote in votes:
        for field in ("model_id", "model_family", "revision", "policy_version"):
            if not isinstance(vote.get(field), str) or not vote[field].strip():
                raise ValueError(f"Missing teacher {field}")
        targets_and_mask(vote["targets"])
        families.add(vote["model_family"].casefold().strip())
        for label, value in vote["targets"].items():
            if value is not None and not str(vote.get("evidence", {}).get(label, "")).strip():
                raise ValueError("Each known teacher target requires an evidence explanation")
    if len(families) != 2:
        raise ValueError("Same model-family votes do not meet this agreement design")
    if votes[0]["policy_version"] != votes[1]["policy_version"]:
        raise ValueError("Teachers used different policies")
    targets = {label: votes[0]["targets"][label]
               if votes[0]["targets"][label] == votes[1]["targets"][label] else None
               for label in LABELS}
    return {"targets": targets, "annotation_kind": "derived_model_consensus",
            "human_boundary_annotation": False, "clinical_ground_truth": False,
            "independence_proven": False,
            "disagreed_labels": [label for label in LABELS
                                 if votes[0]["targets"][label] != votes[1]["targets"][label]]}


def validate_record(row, registry, *, partition):
    for field in ("id", "text", "source", "source_id", "source_revision", "official_split", "root_id", "semantic_family"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"Missing nonempty {field}")
    targets_and_mask(row["targets"])
    if row.get("language") != "en":
        raise ValueError("English language verification required")
    if row.get("policy_version") != "boundary-policy-2.0":
        raise ValueError("Record policy version differs from this design")
    sources = {source["id"]: source for source in registry["sources"]}
    if row["source"] not in sources:
        raise ValueError("Unregistered source")
    source = sources[row["source"]]
    if partition == "train":
        if row["official_split"] not in source["training_partitions"]:
            raise ValueError("Source/official partition not eligible for training")
        if all(value is None for value in row["targets"].values()):
            raise ValueError("Unlabelled candidates cannot enter primary training")
    if partition == "test" and any(row.get(field) is not False for field in
                                   ("previously_trained", "previously_used_for_development")):
        raise ValueError("Exposed or exposure-unknown rows cannot enter a pristine final test")
    if not isinstance(row.get("ancestor_ids"), list) or row["root_id"] not in row["ancestor_ids"]:
        raise ValueError("Complete lineage must include the original root")
    if any(not isinstance(value, str) or not value.strip() for value in row["ancestor_ids"]):
        raise ValueError("Invalid ancestor identity")
    if row.get("ancestry_complete") is not True:
        raise ValueError("Complete ancestry declaration required")
    if row.get("human_boundary_annotation") is not False:
        raise ValueError("Current approved redesign sources do not supply human four-label gold")
    if row.get("clinical_ground_truth") is not False:
        raise ValueError("Clinical ground truth cannot be claimed for these derived targets")
    if row.get("annotation_kind") not in {"derived_model_consensus", "authored_weak", "derived_programmatic"}:
        raise ValueError("Unknown/unapproved custom-target provenance")
    if row["annotation_kind"] == "derived_model_consensus":
        voted = consensus_targets(row.get("teacher_votes", []))
        if row["teacher_votes"][0]["policy_version"] != row["policy_version"]:
            raise ValueError("Teacher policy differs from record policy")
        if voted["targets"] != row["targets"]:
            raise ValueError("Targets differ from recorded teacher agreement")
    if not isinstance(row.get("slices"), list) or not isinstance(row.get("content_origin"), str):
        raise ValueError("Content origin and audit slices required")
    return True


def assert_disjoint(splits):
    """Require every shared lineage/family/source/text key to stay in one split."""
    memberships = {}
    ids = set()
    exact_labels = {}
    for partition, rows in splits.items():
        for row in rows:
            if row["id"] in ids:
                raise ValueError("Duplicate record id")
            ids.add(row["id"])
            normalized = text_key(row["text"])
            key_targets = exact_labels.setdefault(normalized, {})
            for label, value in row["targets"].items():
                if value is not None:
                    if label in key_targets and key_targets[label] != value:
                        raise ValueError("Contradictory exact-text targets")
                    key_targets[label] = value
            keys = [("source_identity", f"{row['source']}:{row['source_id']}"),
                    ("text", normalized), ("family", row["semantic_family"])]
            keys += [("ancestor", value) for value in row["ancestor_ids"]]
            keys += [("document", value) for value in row.get("document_groups", [])]
            keys += [("semantic_cluster", value) for value in row.get("semantic_clusters", [])]
            for key in keys:
                previous = memberships.setdefault(key, partition)
                if previous != partition:
                    raise ValueError(f"Cross-partition lineage/source/text leakage: {key[0]}")
    return {"passed": True, "records": len(ids),
            "semantic_group_quality_independently_validated": False}


def readiness(splits, registry, protocol):
    """Structural readiness, not clinical validation or teacher-quality proof."""
    for partition, rows in splits.items():
        if partition not in protocol["partitions"]:
            raise ValueError("Unknown partition")
        for row in rows:
            validate_record(row, registry, partition=partition)
    audit = assert_disjoint(splits)
    reasons = []
    support = {}
    for partition in protocol["partitions"]:
        rows = splits.get(partition, [])
        if not rows:
            reasons.append(f"{partition}: missing records")
        counts = {}
        for label in LABELS:
            positives = [row for row in rows if row["targets"][label] == 1]
            families = Counter(row["semantic_family"] for row in positives)
            counts[label] = {"positives": len(positives), "families": len(families)}
            needed = protocol["minimum_training_positive_families_per_label"] if partition == "train" else protocol["minimum_evaluation_positive_families_per_label"]
            if len(families) < needed:
                reasons.append(f"{partition}/{label}: insufficient distinct positive families")
            if partition != "train" and len(positives) < protocol["minimum_evaluation_positives_per_label"]:
                reasons.append(f"{partition}/{label}: insufficient positive support")
            if families and max(families.values()) / len(positives) > protocol["maximum_one_family_fraction_of_positive_support"]:
                reasons.append(f"{partition}/{label}: one family dominates positive support")
        benign = sum(target_disposition(row["targets"]) == "allowed" and not row.get("safety_signals") for row in rows)
        if partition != "train" and benign < protocol["minimum_evaluation_fully_known_benign"]:
            reasons.append(f"{partition}: insufficient fully known benign examples")
        missing = sorted(set(protocol["required_slices"]) - {value for row in rows for value in row["slices"]})
        if missing:
            reasons.append(f"{partition}: missing slices {','.join(missing)}")
        support[partition] = {"rows": len(rows), "positive_support": counts, "fully_known_benign": benign}
    train = splits.get("train", [])
    multi = sum(sum(value == 1 for value in row["targets"].values()) > 1 for row in train)
    if multi < protocol["minimum_training_multi_positive_rows"]:
        reasons.append("train: insufficient multi-positive examples")
    positives = [row for row in train if 1 in row["targets"].values()]
    if not positives or all(row["content_origin"] == "authored_template" for row in positives):
        reasons.append("train: positives missing or entirely authored template expansions")
    return {"ready_for_training_data_adapter_review": not reasons, "blocking_reasons": reasons,
            "support": support, "structural_audit": audit, "clinical_validation_established": False}


def freeze_protocol(specs, destination):
    destination = Path(destination)
    hashes = {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in specs}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    payload = {"schema": "boundary-protocol-lock-1.0", "files_sha256": hashes,
               "combined_sha256": digest, "test_content_locked": False,
               "meaning": "Design specification frozen; no new labelled final test set exists yet."}
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
    return payload


def verify_protocol(lock_path):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    hashes = lock["files_sha256"]
    combined = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    if combined != lock["combined_sha256"]:
        raise ValueError("Changed lock file")
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("Protocol changed; create a new version, do not reuse this lock")
    return True
