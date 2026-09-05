"""Stable seeds, provenance and exclusive artifact writes."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260905

def configure_runtime(seed=SEED, threads=4):
    # Official eager inference avoids optional compiler/toolchain dependence on CPU.
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    for key, folder in {'XDG_CACHE_HOME': '.cache', 'TORCH_HOME': '.cache/torch',
                        'HF_HOME': '.cache/huggingface', 'MPLCONFIGDIR': '.cache/matplotlib',
                        'NUMBA_CACHE_DIR': '.cache/numba'}.items():
        os.environ[key] = str(ROOT / folder)
    os.environ['OMP_NUM_THREADS'] = str(threads)
    os.environ['MKL_NUM_THREADS'] = str(threads)
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)

def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:8], 'big')

def message_for(clip_id, seed=SEED):
    return ''.join(map(str, np.random.default_rng(stable_seed(seed, clip_id, 'message')).integers(0, 2, 16)))

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def code_hash():
    digest = hashlib.sha256()
    for folder in ['src', 'scripts', 'configs']:
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()
