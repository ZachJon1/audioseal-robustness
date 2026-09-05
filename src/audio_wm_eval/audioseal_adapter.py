"""Official AudioSeal only; checkpoint identity is verified before loading."""
import json
import time
import numpy as np
from .common import ROOT, sha256
from .audio_io import validate_audio

class AudioSealAdapter:
    def __init__(self, config=None):
        import torch
        from audioseal import AudioSeal
        self.config = config or {}
        self.device = torch.device(self.config.get('device', 'cpu'))
        provenance = json.loads((ROOT / 'outputs/model_provenance.json').read_text())
        paths = {}
        for kind in ('generator', 'detector'):
            entry = provenance[kind]
            path = ROOT / entry['path']
            if sha256(path) != entry['sha256']:
                raise ValueError(f'Official {kind} checkpoint hash mismatch')
            paths[kind] = str(path)
        start = time.perf_counter()
        self.generator = AudioSeal.load_generator(paths['generator'], nbits=16, device=self.device).eval()
        self.detector = AudioSeal.load_detector(paths['detector'], nbits=16, device=self.device).eval()
        self.load_runtime_ms = (time.perf_counter() - start) * 1000

    def tensor(self, audio):
        import torch
        return torch.from_numpy(validate_audio(audio).copy())[None, None, :].to(self.device)

    def sync(self):
        import torch
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def embed(self, audio, message):
        import torch
        if len(message) != 16 or set(message) - {'0', '1'}:
            raise ValueError('Expected 16 binary message characters')
        x = self.tensor(audio)
        msg = torch.tensor([[int(b) for b in message]], device=self.device)
        self.sync()
        start = time.perf_counter()
        with torch.inference_mode():
            wm = self.generator.get_watermark(x, message=msg)
            y = x + self.config.get('watermark_strength', 1.0) * wm
        self.sync()
        elapsed = (time.perf_counter() - start) * 1000
        if y.shape != x.shape:
            raise ValueError('Watermark changed audio tensor shape')
        out = y[0, 0].cpu().numpy()
        # Embedding clipping is a failed gate, never silently change the watermark.
        validate_audio(out)
        return out, elapsed

    def detect(self, audio):
        import torch
        x = self.tensor(audio)
        self.sync()
        start = time.perf_counter()
        with torch.inference_mode():
            frame, bits = self.detector(x)
            p = frame[:, 1, :]
            score = (p > self.config.get('frame_threshold', 0.5)).float().mean().item()
            message = ''.join(map(str, (bits[0] > self.config.get('message_threshold', 0.5)).int().tolist()))
            mean_p = p.mean().item()
        self.sync()
        elapsed = (time.perf_counter() - start) * 1000
        if not 0 <= score <= 1 or len(message) != 16 or not np.isfinite(mean_p):
            raise ValueError('Invalid official detector output')
        return {'detection_score': score, 'mean_frame_probability': mean_p,
                'detected': int(score > self.config.get('detector_threshold', 0.5)),
                'recovered_message': message, 'detect_runtime_ms': elapsed}
