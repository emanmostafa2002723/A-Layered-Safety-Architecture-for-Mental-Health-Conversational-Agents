"""Check source bytes, accidental asset/private files and common secret patterns.

Read-only, model-free. Not a substitute for a human publication/license review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git','.runtime','.venv-api','.venv-demo','.pytest_cache','__pycache__','build','dist'}
ASSET_FOLDERS = ('models/','components/crisis_fresh_epoch3/','components/boundary/model/','rag_project/data/')
ASSET_SUFFIXES = {'.pt','.pth','.safetensors','.bin','.gguf','.npy','.npz','.pkl','.faiss'}
PRIVATE_SUFFIXES = {'.key','.pem','.memory-key','.sqlite3','.db'}
SECRETS = [re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
           re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),
           re.compile(r'github_pat_[A-Za-z0-9_]{40,}')]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_paths(root):
    import os
    for folder, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith(('.venv','.pytest','.runtime'))]
        for name in sorted(names):
            path = Path(folder)/name
            relative = path.relative_to(root).as_posix()
            if path.suffix in {'.pyc','.pyo'} or '.egg-info' in relative:
                continue
            yield path, relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-runtime-assets', action='store_true', help='Audit a server runtime with verified linked assets present')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'config/source_files.json').read_text())
    failures = []
    expected = {row['path']:row for row in manifest['files']}
    checked = 0
    for relative, row in expected.items():
        path = ROOT/relative
        try:
            path.resolve().relative_to(ROOT)
        except ValueError:
            failures.append('source_escape:'+relative)
            continue
        if not path.is_file() or path.is_symlink():
            failures.append('missing_or_linked_source:'+relative)
        elif path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            failures.append('changed_source:'+relative)
        elif path.suffix == '.py':
            try:
                compile(path.read_bytes(),relative,'exec')
            except SyntaxError:
                failures.append('syntax:'+relative)
        checked += 1
    total = 0
    for path, relative in source_paths(ROOT):
        is_asset = relative.startswith(ASSET_FOLDERS) or path.suffix in ASSET_SUFFIXES
        if is_asset:
            if not args.allow_runtime_assets:
                failures.append('asset_not_for_git:'+relative)
            continue
        if relative.startswith('.env') or path.name == 'server-assets.local.json' or path.suffix in PRIVATE_SUFFIXES:
            if args.allow_runtime_assets:
                continue
            failures.append('private_file_not_for_git:'+relative)
            continue
        if path.is_symlink():
            failures.append('source_symlink:'+relative)
            continue
        total += path.stat().st_size
        if relative != 'config/source_files.json' and relative not in expected:
            failures.append('unreviewed_source:'+relative)
        if path.stat().st_size > 20*1024**2:
            failures.append('unexpected_large_source:'+relative)
        if path.suffix in {'.py','.js','.json','.md','.yml','.yaml','.txt','.toml','.ps1','.sh','.html'}:
            text = path.read_text(encoding='utf-8-sig',errors='replace')
            if any(pattern.search(text) for pattern in SECRETS):
                failures.append('possible_secret:'+relative)
    print(json.dumps({'ready':not failures,'source_files_checked':checked,'source_bytes':total,
        'runtime_assets_allowed':args.allow_runtime_assets,'files_changed':False,
        'models_loaded':False,'failures':failures,
        'note':'Source integrity and limited accidental-disclosure checks only; not clinical/license/privacy approval.'},indent=2))
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
