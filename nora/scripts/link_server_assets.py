"""Verify exact assets, then optionally hard-link them into this new checkout.

Default/check mode changes nothing. Does not load models, import old code, copy
large files, change permissions, remove sources, or accept hash substitutions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(8 * 1024**2):
            value.update(block)
    return value.hexdigest()


def destination(root, relative):
    parts = PurePosixPath(relative)
    if parts.is_absolute() or not parts.parts or any(p in {'.','..'} for p in parts.parts) or '\\' in relative or ':' in relative:
        raise ValueError('Unsafe manifest target')
    target = root.joinpath(*parts.parts)
    current = root
    for name in parts.parts:
        current = current/name
        if current.is_symlink():
            raise ValueError('Symlink target is not allowed: ' + relative)
    target.resolve().relative_to(root)
    return target


def plan(root, manifest, sources):
    root = Path(root).resolve()
    rows = []
    if manifest.get('schema_version') != 'nora-runtime-assets/1.0':
        raise ValueError('Unsupported asset manifest')
    seen = set()
    for group, config in manifest['groups'].items():
        for item in config['files']:
            relative = config['target_prefix'] + item['name']
            target = destination(root, relative)
            if relative in seen or not re.fullmatch('[a-f0-9]{64}', item['sha256']) or type(item['bytes']) is not int or item['bytes'] < 1:
                raise ValueError('Invalid asset record')
            seen.add(relative)
            row = {'group': group, 'target': relative}
            if target.exists():
                source = target
                row['status'] = 'already_present'
            elif not sources.get(group):
                row['status'] = 'missing_source_mapping'
                rows.append(row)
                continue
            else:
                base = Path(sources[group]).expanduser()
                if not base.is_absolute():
                    raise ValueError('Source folder must be absolute: ' + group)
                source = base/item['name']
                source.resolve().relative_to(base.resolve())
                row['status'] = 'link_ready'
            row['source'] = str(source)
            if not source.is_file():
                row['status'] = 'missing_file'
            elif source.stat().st_size != item['bytes']:
                row['status'] = 'size_mismatch'
            elif digest(source) != item['sha256']:
                row['status'] = 'hash_mismatch'
            elif row['status'] == 'link_ready' and source.stat().st_dev != root.stat().st_dev:
                row['status'] = 'different_filesystem_no_automatic_copy'
            rows.append(row)
    return rows


def apply(root, rows):
    if any(row['status'] not in {'link_ready','already_present'} for row in rows):
        raise ValueError('Not all assets verified; nothing linked')
    created = []
    try:
        for row in rows:
            if row['status'] != 'link_ready':
                continue
            target = destination(root, row['target'])
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(row['source'], target)
            created.append(target)
    except Exception:
        # Roll back only hard links created by this invocation; source untouched.
        for target in reversed(created):
            if target.is_file():
                target.unlink()
        raise
    return len(created)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', type=Path, help='Group -> absolute source-folder JSON')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Read-only (default)')
    mode.add_argument('--apply', action='store_true', help='Create links only after all hashes pass')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'config/runtime_assets.json').read_text())
    sources = json.loads(args.sources.read_text()) if args.sources else {}
    try:
        rows = plan(ROOT, manifest, sources)
        ready = all(row['status'] in {'link_ready','already_present'} for row in rows)
        linked = apply(ROOT, rows) if args.apply else 0
        print(json.dumps({'ready': ready, 'mode': 'apply' if args.apply else 'check',
            'links_created': linked, 'model_loaded': False, 'files': rows}, indent=2))
        return 0 if ready else 2
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({'ready': False, 'error': str(error), 'models_loaded': False}))
        return 2


if __name__ == '__main__':
    sys.exit(main())
