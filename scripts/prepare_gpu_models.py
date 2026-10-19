"""Pinned course assets and public model cache, with a persistent preparation manifest.

Run on the GPU instance after scripts/setup_gpu_instance.sh succeeds.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import time

from huggingface_hub import snapshot_download

from common.data import REPO_ROOT, load_yaml
from scripts.download_assets import HF_ASSET_REVISION, main as download_assets


def stamp(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    logs = REPO_ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    state = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
             'course_revision': HF_ASSET_REVISION, 'stage': 'course_assets', 'models': {}}
    output = logs / 'prepare_models_state.json'
    save = lambda: output.write_text(json.dumps(state, indent=2), encoding='utf-8')
    start = time.perf_counter()
    try:
        save()
        stamp('Downloading fixed course assets')
        download_assets()
        expected_path = REPO_ROOT / 'manifests/reviewed_asset_hashes.json'
        if expected_path.exists():
            expected = json.loads(expected_path.read_text())
            actual = {name: sha256(REPO_ROOT / name) for name in expected}
            if actual != expected:
                raise RuntimeError('Downloaded asset hashes differ from independently reviewed files')
            (logs / 'verified_asset_hashes.json').write_text(json.dumps(actual, indent=2))
            stamp(f'Independently reviewed asset hashes match: {len(actual)} files')
        state['stage'] = 'asset_validation'; save()
        subprocess.run([sys.executable, '-m', 'scripts.validate_assets'], cwd=REPO_ROOT, check=True)
        cfg = load_yaml('configs/base.yaml')
        for role in ['base_model', 'reward_model', 'ai_judge_model']:
            model_id = cfg[role]
            state['stage'] = role; save()
            stamp(f'Downloading public model {model_id}')
            tick = time.perf_counter()
            # Download main once and record its resolved immutable snapshot. Subsequent
            # validation runs use HF_HUB_OFFLINE to keep this exact cached revision.
            path = Path(snapshot_download(model_id, allow_patterns=[
                '*.json', '*.safetensors', '*.model', '*.txt', '*.jinja',
            ], max_workers=4))
            generation = path / 'generation_config.json'
            state['models'][role] = {
                'repo_id': model_id, 'revision': path.name, 'snapshot_path': str(path),
                'download_wall_s': time.perf_counter() - tick,
                'generation_config': json.loads(generation.read_text()) if generation.exists() else None,
                'files': {p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)}
                          for p in path.iterdir() if p.is_file()},
            }
            save()
            stamp(f'{model_id}: cached and hashed at revision {path.name}')
        state['packages'] = {name: metadata.version(name) for name in [
            'torch', 'transformers', 'tokenizers', 'peft', 'trl', 'datasets',
            'accelerate', 'bitsandbytes', 'huggingface_hub', 'numpy', 'pandas',
        ]}
        state.update(status='passed', stage='complete')
    except Exception as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state.update(updated_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter() - start)
        save()


if __name__ == '__main__':
    main()
