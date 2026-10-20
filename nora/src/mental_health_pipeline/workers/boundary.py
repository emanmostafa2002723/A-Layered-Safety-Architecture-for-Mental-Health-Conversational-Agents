"""Persistent boundary V11/V12 worker for its locked Python environment."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import sys

from mental_health_pipeline.guardrails import LazyBoundaryV11Adapter
from mental_health_pipeline.workers.common import announce_startup_error, serve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            adapter = LazyBoundaryV11Adapter(
                component_root=args.component_root,
                config_path=args.config,
                device=args.device,
            )
            adapter._load()
    except Exception:
        announce_startup_error("boundary")
        return 2
    return serve("boundary", "predict", adapter.predict)


if __name__ == "__main__":
    raise SystemExit(main())
