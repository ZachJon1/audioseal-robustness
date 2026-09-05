"""Download only immutable official example and checkpoint URLs (<100 MB each)."""
import json
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audio_wm_eval.common import ROOT, sha256, write_json

SOURCE_REVISION = 'e63a8a0e5cdf7bb797159c92ba15961557fe9bd2'
MODEL_REVISION = '3c19eba53390776cf2cc9ed5f6c9ac67ce72ecba'

def main():
    target = ROOT / 'outputs/model_provenance.json'
    if target.exists():
        for item in json.loads(target.read_text()).values():
            if isinstance(item, dict) and 'path' in item:
                assert sha256(ROOT / item['path']) == item['sha256']
        print('Existing official artifacts verified')
        return
    entries = {'source_revision': SOURCE_REVISION, 'model_revision': MODEL_REVISION}
    for kind, url, rel in [
        ('generator', f'https://huggingface.co/facebook/audioseal/resolve/{MODEL_REVISION}/generator_base.pth', 'checkpoints/generator_base.pth'),
        ('detector', f'https://huggingface.co/facebook/audioseal/resolve/{MODEL_REVISION}/detector_base.pth', 'checkpoints/detector_base.pth'),
        ('example', f'https://raw.githubusercontent.com/facebookresearch/audioseal/{SOURCE_REVISION}/test.wav', 'data/raw/official_test.wav')]:
        path = ROOT / rel
        path.parent.mkdir(exist_ok=True, parents=True)
        with urllib.request.urlopen(url, timeout=120) as response, path.open('xb') as out:
            total = 0
            for block in iter(lambda: response.read(1024 * 1024), b''):
                total += len(block)
                if total > 100_000_000:
                    raise RuntimeError('Unexpected download over declared 100 MB artifact budget')
                out.write(block)
        entries[kind] = {'url': url, 'path': rel, 'bytes': total, 'sha256': sha256(path)}
        print(kind, total, entries[kind]['sha256'], flush=True)
    write_json(target, entries)

if __name__ == '__main__':
    main()
