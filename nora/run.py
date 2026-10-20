"""Portable private launcher. Model assets and dependencies must be restored first."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['web','gradio','preflight','rag-check','qwen-check','smoke'])
    parser.add_argument('--env-file', type=Path, default=ROOT/'.env.server')
    parser.add_argument('--port', type=int)
    parser.add_argument('--message', default='Hi')
    args = parser.parse_args()
    env_file = args.env_file if args.env_file.is_absolute() else ROOT/args.env_file
    if not env_file.is_file():
        parser.error('Copy config/server-assets.env.example to .env.server first')
    sys.path.insert(0, str(ROOT/'src'))
    from mental_health_pipeline.cli import load_env_file
    load_env_file(env_file)
    for key in ('MHP_CRISIS_PYTHON','MHP_BOUNDARY_PYTHON','MHP_OUTPUT_GUARD_PYTHON'):
        os.environ.setdefault(key, sys.executable)
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT/'src')
    env['GRADIO_ANALYTICS_ENABLED'] = 'False'
    if args.command == 'web':
        cmd = [sys.executable,str(ROOT/'scripts/run_account_memory.py'),'--env-file',str(env_file),'--port',str(args.port or 7861)]
    elif args.command == 'gradio':
        cmd = [sys.executable,'-m','mental_health_pipeline.gradio_app','--project-root',str(ROOT),'--env-file',str(env_file),'--port',str(args.port or 7860)]
    elif args.command == 'qwen-check':
        # Check the selected one-field router, not the older three-field V4 CLI probe.
        from mental_health_pipeline.config import PipelineConfig
        from mental_health_pipeline.minicpm_router import build_minicpm_router
        import json
        config = PipelineConfig.from_env(project_root=ROOT)
        router = build_minicpm_router(config)
        error = router.readiness_code()
        print(json.dumps({'ready':error is None,'error_code':error,'router_backend':config.rag_router_backend,
            'model':router.model,'inference_tested':False},indent=2))
        return int(error is not None)
    else:
        cmd = [sys.executable,'-m','mental_health_pipeline.cli','--project-root',str(ROOT),'--env-file',str(env_file),args.command]
        if args.command == 'smoke':
            cmd += [args.message,'--adult-confirmed']
    try:
        return subprocess.call(cmd, cwd=ROOT, env=env)
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
