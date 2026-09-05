"""Controlled, deterministic transformations of mono floating-point audio.

All attacks return audio at the original sample rate. A final hard limiter keeps
samples in [-1, 1]; its use is recorded separately from full-scale sample rate.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _audio_array(audio: np.ndarray) -> np.ndarray:
    array = np.asarray(audio)
    if array.ndim != 1 or not array.size:
        raise ValueError("Audio must be a nonempty mono array with shape [T].")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError("Audio must contain floating-point samples.")
    if not np.isfinite(array).all():
        raise ValueError("Audio must contain only finite samples.")
    if np.max(np.abs(array)) > 1.0:
        raise ValueError("Input audio samples must lie in [-1, 1].")
    return np.array(array, dtype=np.float32, copy=True)


def _snr(reference: np.ndarray, difference: np.ndarray) -> float | None:
    signal_power = float(np.mean(np.square(reference, dtype=np.float64)))
    noise_power = float(np.mean(np.square(difference, dtype=np.float64)))
    if signal_power == 0:
        return None
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def _mp3_roundtrip(
    audio: np.ndarray, sample_rate: int, bitrate: int, ffmpeg_path: str | None
) -> np.ndarray:
    if ffmpeg_path is None:
        import imageio_ffmpeg

        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory(prefix="audioseal-mp3-") as temporary:
        directory = Path(temporary)
        source = directory / "input.wav"
        encoded = directory / "encoded.mp3"
        decoded = directory / "decoded.wav"
        sf.write(source, audio, sample_rate, subtype="FLOAT")
        commands = [
            [
                ffmpeg_path, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-y", "-i", str(source), "-map_metadata", "-1", "-ac", "1",
                "-ar", str(sample_rate), "-c:a", "libmp3lame", "-b:a",
                f"{bitrate}k", "-threads", "1", str(encoded),
            ],
            [
                ffmpeg_path, "-nostdin", "-hide_banner", "-loglevel", "error",
                "-y", "-i", str(encoded), "-ac", "1", "-ar", str(sample_rate),
                "-c:a", "pcm_f32le", "-threads", "1", str(decoded),
            ],
        ]
        for command in commands:
            try:
                subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"FFmpeg MP3 round-trip failed: {error.stderr.strip()}") from error
        result, decoded_rate = sf.read(decoded, dtype="float32", always_2d=False)
        if decoded_rate != sample_rate or result.ndim != 1:
            raise RuntimeError("FFmpeg returned an unexpected sample rate or channel layout.")
        return result


def apply_attack(
    audio: np.ndarray,
    sample_rate: int,
    condition: dict[str, Any],
    seed: int,
    ffmpeg_path: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply one configuration-defined attack without modifying the input.

    ``noise_snr_db_pre_clipping`` is the exact configured Gaussian-noise SNR.
    ``noise_snr_db`` is measured after float32 conversion and limiting. Noise
    realizations and crop start points match across branches when seeds match.
    MP3, pitch, stretch and crop conservatively disallow sample-aligned metrics.
    """
    source = _audio_array(audio)
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    family = condition.get("family")
    metadata: dict[str, Any] = {
        **condition,
        "seed": int(seed),
        "sample_rate": int(sample_rate),
        "input_samples": len(source),
        "sample_aligned": family in {"clean", "noise", "resample"},
    }
    generator = np.random.default_rng(seed)
    if family == "clean":
        transformed = source.copy()
    elif family == "noise":
        snr_db = float(condition["snr_db"])
        if not np.isfinite(snr_db):
            raise ValueError("snr_db must be finite.")
        signal_power = float(np.mean(np.square(source, dtype=np.float64)))
        if signal_power == 0:
            raise ValueError("A target noise SNR is undefined for silent input.")
        noise = generator.standard_normal(source.size)
        noise_power = float(np.mean(noise**2))
        noise *= np.sqrt(signal_power / noise_power) * 10 ** (-snr_db / 20)
        transformed = source.astype(np.float64) + noise
        metadata["noise_snr_db_pre_clipping"] = _snr(source, noise)
    elif family == "mp3":
        bitrate = int(condition["bitrate_kbps"])
        if bitrate not in {64, 128}:
            raise ValueError("The baseline MP3 bitrate must be 64 or 128 kbps.")
        transformed = _mp3_roundtrip(source, sample_rate, bitrate, ffmpeg_path)
        metadata["codec"] = "libmp3lame"
        metadata["alignment_reason"] = "MP3 codec delay/alignment not independently verified"
    elif family == "resample":
        intermediate_rate = int(condition["intermediate_sr"])
        if not 0 < intermediate_rate < sample_rate:
            raise ValueError("intermediate_sr must be positive and below sample_rate.")
        divisor = math.gcd(sample_rate, intermediate_rate)
        lower_rate = resample_poly(source, intermediate_rate // divisor, sample_rate // divisor)
        transformed = resample_poly(lower_rate, sample_rate // divisor, intermediate_rate // divisor)
        transformed = transformed[: source.size]
        if transformed.size != source.size:
            raise RuntimeError("Resample round-trip unexpectedly shortened the signal.")
        metadata["resampler"] = "scipy.signal.resample_poly"
        metadata["resampler_window"] = ["kaiser", 5.0]
    elif family in {"pitch", "stretch"}:
        import librosa

        n_fft = int(condition.get("n_fft", 2048))
        hop_length = int(condition.get("hop_length", n_fft // 4))
        if n_fft < 2 or not 0 < hop_length <= n_fft:
            raise ValueError("Require n_fft >= 2 and 0 < hop_length <= n_fft.")
        metadata.update({"n_fft": n_fft, "hop_length": hop_length, "implementation": "librosa"})
        if family == "pitch":
            semitones = float(condition["semitones"])
            if not np.isfinite(semitones):
                raise ValueError("semitones must be finite.")
            transformed = librosa.effects.pitch_shift(
                source, sr=sample_rate, n_steps=semitones,
                bins_per_octave=12, res_type="soxr_hq", n_fft=n_fft, hop_length=hop_length,
            )
            metadata["res_type"] = "soxr_hq"
            metadata["alignment_reason"] = "Pitch shifting changes sample phase and content"
        else:
            rate = float(condition["rate"])
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("stretch rate must be positive and finite.")
            transformed = librosa.effects.time_stretch(
                source, rate=rate, n_fft=n_fft, hop_length=hop_length,
            )
            metadata["alignment_reason"] = "Time stretching changes the temporal coordinate"
    elif family == "crop":
        fraction = float(condition["fraction"])
        if not np.isfinite(fraction) or not 0 <= fraction < 1:
            raise ValueError("crop fraction must be in [0, 1).")
        retained = max(1, int(round(source.size * (1 - fraction))))
        start = int(generator.integers(0, source.size - retained + 1))
        transformed = source[start : start + retained].copy()
        metadata.update({
            "crop_start_sample": start,
            "crop_end_sample": start + retained,
            "removed_fraction_actual": 1 - retained / source.size,
            "alignment_reason": "Contiguous crop changes length and may shift time origin",
        })
    else:
        raise ValueError(f"Unsupported attack family: {family!r}")
    transformed = np.asarray(transformed)
    if transformed.ndim != 1 or not transformed.size or not np.isfinite(transformed).all():
        raise RuntimeError("Attack produced empty, non-mono, or non-finite audio.")
    metadata["clipping_fraction"] = float(np.mean(np.abs(transformed) > 1.0))
    metadata["pre_limiter_peak"] = float(np.max(np.abs(transformed)))
    output = np.clip(transformed, -1, 1).astype(np.float32)
    metadata["output_peak"] = float(np.max(np.abs(output)))
    metadata["output_samples"] = len(output)
    metadata["duration_seconds"] = len(output) / sample_rate
    if family == "noise":
        metadata["noise_snr_db"] = _snr(source, output.astype(np.float64) - source)
    return output, metadata
