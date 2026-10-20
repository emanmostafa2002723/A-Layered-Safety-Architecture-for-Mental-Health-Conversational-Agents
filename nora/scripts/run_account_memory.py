"""Launch the private account/memory web UI over the unchanged full safety pipeline."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--env-file', type=Path, default=ROOT/'.env.server')
    parser.add_argument('--init-only', action='store_true', help='Create empty private database, without model loading or serving')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('port must be 1024..65535')
    from cryptography.fernet import Fernet
    from mental_health_pipeline.account_store import AccountStore
    private = ROOT/'.runtime'/'private-accounts'
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    keyfile, database = private/'accounts.memory-key', private/'accounts.sqlite3'
    if not keyfile.exists():
        if database.exists():
            raise SystemExit('Encryption key missing for existing database. Restore its key; do not replace it.')
        # Exclusive create; never overwrite an existing key or print it.
        fd = os.open(keyfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(Fernet.generate_key())
    store = AccountStore(database, encryption_key=keyfile.read_bytes().strip())
    if args.init_only:
        store.purge()
        store.close()
        print('Empty/private account store ready. No model loaded. No server started.')
        print('Private data directory (never upload):', private)
        return
    from mental_health_pipeline.cli import load_env_file
    from mental_health_pipeline.config import PipelineConfig, RuntimeMode
    from mental_health_pipeline.bootstrap import build_pipeline
    from mental_health_pipeline.account_service import AccountChatService
    from mental_health_pipeline.account_api import AccountWebSettings, create_account_app
    import uvicorn
    env_file = args.env_file if args.env_file.is_absolute() else ROOT/args.env_file
    if not env_file.is_file():
        raise SystemExit('Missing .env.server. Copy config/server-assets.env.example to .env.server first.')
    load_env_file(env_file)
    # A reviewed project-local environment may host all three workers.
    # Explicit per-worker MHP_*_PYTHON overrides remain supported.
    for name in ('MHP_CRISIS_PYTHON','MHP_BOUNDARY_PYTHON','MHP_OUTPUT_GUARD_PYTHON'):
        os.environ.setdefault(name, sys.executable)
    config = PipelineConfig.from_env(project_root=ROOT)
    if config.runtime_mode is RuntimeMode.PRODUCTION:
        raise SystemExit('Account prototype is not approved for public production deployment.')
    if config.output_guard_backend != 'local_qwen':
        raise SystemExit('This prototype requires the selected full Qwen output guard, not a placeholder.')
    import torch
    torch.set_num_threads(4)
    pipeline = build_pipeline(config)
    try:
        origin = f'http://127.0.0.1:{args.port}'
        app = create_account_app(AccountChatService(pipeline, store), AccountWebSettings(origin))
        print('Open:', origin)
        print('Private test only. Stop any other model UI first; this starts its own full pipeline.')
        uvicorn.run(app, host='127.0.0.1', port=args.port, workers=1, access_log=False)
    finally:
        pipeline.close()
        store.close()


if __name__ == '__main__':
    main()
