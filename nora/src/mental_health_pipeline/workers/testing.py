"""Tiny protocol fixture used only by the integration test suite."""

from __future__ import annotations

import argparse

from mental_health_pipeline.workers.common import serve


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("crisis", "boundary"), required=True)
    args = parser.parse_args()
    if args.component == "crisis":
        return serve("crisis", "assess", _crisis)
    return serve("boundary", "predict", _boundary)


def _crisis(_text: str):
    return {
        "tier": "normal",
        "primary_category": "no_crisis",
        "categories": [],
        "temporal_status": "unspecified",
        "severity": "none",
        "topic_tags": [],
        "classifier": {
            "status": "ok", "predicted_labels": [], "near_threshold_labels": []
        },
        "rule_trace": {"rule_channel": {"matches": []}},
    }


def _boundary(_text: str):
    return {
        "status": "ok",
        "suggested_route": "allow",
        "labels": [],
        "needs_review": False,
        "input_coverage": {"complete_input_seen": True},
        "explicit_emergency_screen": {"explicit_signal": False},
    }


if __name__ == "__main__":
    raise SystemExit(main())
