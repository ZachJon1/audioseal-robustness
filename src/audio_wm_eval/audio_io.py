"""Canonical mono 16 kHz audio with explicit range checks."""
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

def validate_audio(audio, sample_rate=16000):
    a = np.asarray(audio)
    if sample_rate != 16000 or a.ndim != 1 or a.size < 1:
        raise ValueError('Expected nonempty mono [samples] audio at 16000 Hz')
    if not np.isfinite(a).all() or np.max(np.abs(a)) > 1.0:
        raise ValueError('Audio must be finite and bounded by [-1,1]')
    return np.ascontiguousarray(a, dtype=np.float32)

def load_audio(path, convert=False):
    a, sr = sf.read(path, dtype='float32', always_2d=True)
    if not convert and (a.shape[1] != 1 or sr != 16000):
        raise ValueError('Input is not canonical mono 16kHz')
    a = a.mean(axis=1)
    if sr != 16000:
        g = gcd(sr, 16000)
        a = resample_poly(a, 16000 // g, sr // g).astype(np.float32)
    return validate_audio(a), 16000

def save_audio(path, audio):
    a = validate_audio(audio)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        from scipy.io import wavfile
        wavfile.write(f, 16000, a)
