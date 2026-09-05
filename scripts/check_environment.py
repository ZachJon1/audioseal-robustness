"""Capture resolved software, hardware and model provenance without credentials."""
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audio_wm_eval.common import ROOT, configure_runtime, write_json, sha256
configure_runtime()
import imageio_ffmpeg
import torch

def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return {'command': args, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': args, 'error': str(exc)}

def main():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = {'python': sys.version, 'python_executable': sys.executable,
              'platform': platform.platform(), 'os_release': platform.freedesktop_os_release(),
              'cpu_count': os.cpu_count(), 'cpu': command(['lscpu']),
              'memory': command(['free', '-b']), 'disk': dict(zip(('total','used','free'), shutil.disk_usage(ROOT))),
              'nvidia_smi': command(['nvidia-smi']), 'cuda_available': torch.cuda.is_available(),
              'torch_cuda_version': torch.version.cuda, 'device': 'cpu', 'torch_threads': torch.get_num_threads(),
              'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
              'torch_compile_disabled': os.environ.get('TORCHDYNAMO_DISABLE') == '1',
              'seed': 20260905, 'ffmpeg_path': ffmpeg, 'ffmpeg_sha256': sha256(ffmpeg),
              'ffmpeg': command([ffmpeg, '-version']), 'git': command(['git', '--version']),
              'packages': {d.metadata['Name']: d.version for d in sorted(importlib.metadata.distributions(), key=lambda d:d.metadata['Name'].lower())},
              'lock_sha256': sha256(ROOT/'requirements-lock.txt')}
    for name in ('model_provenance', 'checkpoint_source_metadata', 'audioseal_source_commit'):
        path = ROOT/f'outputs/{name}.json'
        if path.exists():
            result[name] = json.loads(path.read_text())
    write_json(ROOT/'outputs/environment.json', result)
    print(json.dumps({'python': result['python'], 'cuda_available': result['cuda_available'],
                      'packages': len(result['packages']), 'ffmpeg': result['ffmpeg']['stdout'].splitlines()[0]}, indent=2))

if __name__ == '__main__':
    main()
