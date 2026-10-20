"""Persistent crisis-classifier worker for its locked Python environment."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import sys
from pathlib import Path

from mental_health_pipeline.guardrails import LazyStaticCrisisAdapter
from mental_health_pipeline.new_components import S600CrisisAdapter
from mental_health_pipeline.fresh_crisis import FreshCrisisAdapter
from mental_health_pipeline.workers.common import announce_startup_error, serve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expected-model-sha256")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--backend", choices=('fresh_epoch3', 's600'), required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--calibration-sha256")
    parser.add_argument("--threads",type=int,default=4)
    args = parser.parse_args()
    try:
        # Reserve stdout exclusively for the protocol. Third-party model-load
        # progress is redirected to stderr, which the client safely drains.
        with redirect_stdout(sys.stderr):
            import torch
            if args.threads < 1:
                raise ValueError('Invalid CPU thread count')
            torch.set_num_threads(args.threads)
            if args.backend == 'fresh_epoch3':
                adapter = FreshCrisisAdapter(component_root=Path(args.weights).parent,
                    weights=args.weights, calibration=args.calibration,
                    weights_sha256=args.weights_sha256,
                    calibration_sha256=args.calibration_sha256, device=args.device)
            else:
                adapter = S600CrisisAdapter(
                    weights=args.weights, weights_sha256=args.weights_sha256,
                    component_root=args.component_root,
                    checkpoint=args.checkpoint,
                    device=args.device,
                    expected_model_sha256=args.expected_model_sha256,
                )
            adapter._load()
    except Exception:
        announce_startup_error("crisis")
        return 2
    return serve("crisis", "assess", adapter.assess)


if __name__ == "__main__":
    raise SystemExit(main())
