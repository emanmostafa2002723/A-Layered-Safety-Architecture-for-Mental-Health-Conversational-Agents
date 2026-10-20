"""Read-only server asset inventory; no model imports, installs or file changes."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def main():
    home = Path.home()
    old = home/'full_pipeline_server/Mental_Health_Chat_Full_Pipeline_ST01c_PhoenixV3_2026-10-14'
    expected = {
        'adapter_model.safetensors': ('phoenix_v6', '7742bb3438fb53c34862d88c031d2f8e5da2f0ceaff52c693804d570a9769da0'),
        'epoch_3.pt': ('crisis_epoch3', '82bb3befdebd3311833a14b299a921cbd120764f6b9dd75766fd61112c61bf68'),
        'calibration.json': ('crisis_calibration', '969101aee690025edfec12344b5f5980c3fb66a1b863ec50d87e4c4de9d7493d'),
    }
    skip = {'node_modules', 'site-packages', '__pycache__', 'venv', 'env',
            'logs', 'private-accounts', 'conversations', 'wandb'}
    models, rag, code, checks, links, inaccessible = {}, {}, [], [], [], []
    roots = [home, Path(os.environ.get('HF_HUB_CACHE', str(home/'.cache/huggingface/hub')))]
    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        roots.append(Path(hf_home)/'hub')
    seen, hash_counts = set(), {}
    def origin(path):
        return 'OLDER_INSTALLATION_NOT_CURRENT_CODE' if path == old or old in path.parents else 'OTHER_LOCATION_NOT_YET_VERIFIED'
    def command(args):
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=12)
            return (result.stdout or result.stderr).strip()[:12000]
        except (OSError, subprocess.TimeoutExpired) as error:
            return type(error).__name__
    for base in dict.fromkeys(roots):
        if not base.is_dir():
            continue
        for folder, dirs, names in os.walk(base, followlinks=False,
                onerror=lambda error: inaccessible.append(str(error.filename))):
            path = Path(folder)
            if str(path) in seen:
                dirs[:] = []
                continue
            seen.add(str(path))
            if len(path.relative_to(base).parts) > 12:
                dirs[:] = []
                continue
            dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d not in skip)
            for directory in dirs:
                candidate = path/directory
                if candidate.is_symlink():
                    links.append({'path': str(candidate), 'target': os.readlink(candidate)})
            if 'src' in dirs and (path/'src/mental_health_pipeline').is_dir():
                code.append({'path': str(path), 'origin': origin(path),
                    'connected_nora_files_present': (path/'src/mental_health_pipeline/web_nora/connection.js').is_file()})
            weights = [n for n in names if Path(n).suffix.lower() in {'.pt', '.pth', '.bin', '.safetensors', '.gguf'}]
            indexes = [n for n in names if Path(n).suffix.lower() in {'.faiss', '.pkl', '.npy', '.npz'}
                or (Path(n).suffix.lower() in {'.json', '.jsonl'} and
                    any(word in n.lower() for word in ('chunks', 'corpus', 'documents', 'index', 'bm25')))]
            if weights:
                models[str(path)] = {'origin': origin(path), 'files': sorted(weights),
                    'configs_present': [n for n in ('config.json', 'adapter_config.json', 'tokenizer.json',
                        'tokenizer_config.json', 'calibration.json') if n in names]}
            if indexes:
                rag[str(path)] = sorted(indexes)
            for name in names:
                if name not in expected:
                    continue
                label, wanted = expected[name]
                if ('phoenix_v6' not in str(path).lower() if label == 'phoenix_v6' else 'crisis' not in str(path).lower()):
                    continue
                if hash_counts.get(label, 0) >= 8:
                    continue
                hash_counts[label] = hash_counts.get(label, 0)+1
                candidate = path/name
                try:
                    digest = hashlib.sha256()
                    with candidate.open('rb') as handle:
                        while block := handle.read(8*1024**2):
                            digest.update(block)
                    checks.append({'path': str(candidate), 'asset': label,
                        'bytes': candidate.stat().st_size, 'sha256': digest.hexdigest(),
                        'matches_current_local_release': digest.hexdigest() == wanted, 'origin': origin(path)})
                except OSError:
                    inaccessible.append(str(candidate))
    disk = shutil.disk_usage(home)
    print(json.dumps({'inventory_only': True, 'files_changed': False,
        'current_code_source': 'LATEST_LOCAL_NORA_TO_BE_UPLOADED_NOT_THE_OLD_SERVER_COPY',
        'older_pipeline_path': str(old), 'older_pipeline_exists': old.is_dir(),
        'home': str(home), 'os': platform.system(), 'python': platform.python_version(),
        'free_disk_gib': round(disk.free/1024**3, 2), 'scan_max_depth': 12,
        'hash_limit_per_current_asset': 8, 'code_locations': code, 'model_folders': models,
        'rag_asset_folders': rag, 'current_release_hash_checks': checks, 'directory_links': links,
        'inaccessible_paths_count': len(inaccessible),
        'gpu': command(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,utilization.gpu', '--format=csv']),
        'ollama_models': command(['ollama', 'list']), 'ollama_loaded': command(['ollama', 'ps']),
        'ollama_models_environment_path': os.environ.get('OLLAMA_MODELS'),
        'scope': 'Home and named Hugging Face cache roots only. Hidden/private directories and virtual environments are skipped. Files outside these roots may exist. No .env, chat content, passwords or model tensors are deserialized.'}, indent=2))


if __name__ == '__main__':
    main()
