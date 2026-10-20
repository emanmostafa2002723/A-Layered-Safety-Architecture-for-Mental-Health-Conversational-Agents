"""Build a private classifier-only ZIP from a verified complete local project.

Not for GitHub. Does not modify input files or include training data, old model
versions, base Mistral/Qwen weights, environments, accounts or RAG text.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.from_project.resolve()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents or output.exists():
        parser.error('Output must be a new ZIP outside this source checkout')
    manifest = json.loads((ROOT/'config/runtime_assets.json').read_text())
    files = []
    for key in ('crisis','boundary'):
        group = manifest['groups'][key]
        for item in group['files']:
            relative = group['target_prefix'] + item['name']
            path = source/relative
            path.resolve().relative_to(source)
            if not path.is_file() or path.stat().st_size != item['bytes']:
                raise ValueError('Missing/incorrect asset: '+relative)
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                while block := handle.read(8*1024**2):
                    digest.update(block)
            if digest.hexdigest() != item['sha256']:
                raise ValueError('Hash mismatch: '+relative)
            files.append((path,relative))
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for path, relative in files:
            archive.write(path,relative)
    digest = hashlib.sha256()
    with output.open('rb') as handle:
        while block := handle.read(8*1024**2):
            digest.update(block)
    print(json.dumps({'private_server_asset_bundle':str(output),'files':len(files),
        'bytes':output.stat().st_size,'sha256':digest.hexdigest(),'for_github':False},indent=2))


if __name__ == '__main__':
    main()
