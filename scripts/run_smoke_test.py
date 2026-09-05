"""One official test.wav; real pretrained positive and unwatermarked inference."""
import json
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from audio_wm_eval.common import ROOT, configure_runtime, message_for, write_json
configure_runtime()
from audio_wm_eval.audio_io import load_audio, save_audio
from audio_wm_eval.audioseal_adapter import AudioSealAdapter
import numpy as np

def main():
    output = ROOT / 'outputs/smoke_test.json'
    if output.exists():
        raise FileExistsError('Refusing to overwrite smoke test evidence')
    result = {'status': 'failed', 'seed': 20260905, 'source': 'official AudioSeal test.wav'}
    try:
        x, sr = load_audio(ROOT / 'data/raw/official_test.wav', convert=True)
        message = message_for('official_test')
        adapter = AudioSealAdapter()
        y, elapsed = adapter.embed(x, message)
        pos = adapter.detect(y)
        neg = adapter.detect(x)
        error = sum(a != b for a, b in zip(message, pos['recovered_message']))
        result.update(sample_rate=sr, tensor_shape=[1, 1, len(x)], duration_seconds=len(x)/sr,
                      expected_message=message, positive=pos, negative=neg, bit_errors=error,
                      bit_accuracy=1-error/16, embed_runtime_ms=elapsed,
                      watermark_snr_db=float(10*np.log10(np.sum(x.astype(float)**2)/np.sum((y.astype(float)-x)**2))),
                      peak=float(np.max(np.abs(y))), model_load_runtime_ms=adapter.load_runtime_ms)
        assert pos['detected'] == 1, 'Clean positive not detected'
        assert error == 0, 'Clean smoke message not exactly recovered'
        assert neg['detected'] == 0, 'Official example negative false positive needs inspection'
        # Cross-check our transparent score calculation against the official high-level API.
        import torch
        with torch.inference_mode():
            score, bits = adapter.detector.detect_watermark(adapter.tensor(y))
        assert abs(score.item()-pos['detection_score']) < 1e-7
        assert ''.join(map(str, bits[0].tolist())) == pos['recovered_message']
        save_audio(ROOT / 'outputs/audio_examples/smoke_positive.wav', y)
        save_audio(ROOT / 'outputs/audio_examples/smoke_negative.wav', x)
        result['status'] = 'passed'
    except Exception as exc:
        result.update(error_message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_json(output, result)
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
