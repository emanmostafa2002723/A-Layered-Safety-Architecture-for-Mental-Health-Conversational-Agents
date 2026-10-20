"""Isolated selected boundary loader, using the existing JSONL protocol."""
import argparse
from contextlib import redirect_stdout
import sys
from mental_health_pipeline.new_components import FinalBoundaryAdapter
from mental_health_pipeline.workers.common import announce_startup_error, serve


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--component-root', required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    try:
        with redirect_stdout(sys.stderr):
            adapter = FinalBoundaryAdapter(a.component_root, a.device)
            adapter._load()
    except Exception:
        announce_startup_error('boundary')
        return 2
    return serve('boundary', 'predict', adapter.predict)


if __name__ == '__main__':
    raise SystemExit(main())
