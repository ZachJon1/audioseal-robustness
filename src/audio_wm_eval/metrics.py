"""Message, audio-quality and clip-level statistical metrics."""

from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np


def compare_messages(expected: str, recovered: str) -> dict[str, int | float | bool]:
    """Compare two exactly 16-bit strings; malformed output is not scored."""
    for name, message in (("expected", expected), ("recovered", recovered)):
        if not isinstance(message, str) or len(message) != 16 or set(message) - {"0", "1"}:
            raise ValueError(f"{name} must be an exactly 16-character binary string.")
    errors = sum(left != right for left, right in zip(expected, recovered))
    return {
        "bit_errors": errors,
        "bit_accuracy": (16 - errors) / 16,
        "bit_error_rate": errors / 16,
        "exact_message_recovery": errors == 0,
    }


def _mono_finite(audio: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(audio, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError(f"{name} must be nonempty, mono, and finite.")
    return values


def aligned_snr(
    reference: np.ndarray, degraded: np.ndarray, sample_aligned: bool = True
) -> float | None:
    """Return sample-aligned SNR in dB, or None when it is inapplicable.

    Different lengths, unverified alignment and zero reference energy are
    inapplicable. Identical non-silent signals have positive infinite SNR.
    """
    original = _mono_finite(reference, "reference")
    changed = _mono_finite(degraded, "degraded")
    if not sample_aligned or original.shape != changed.shape:
        return None
    signal_power = float(np.mean(original**2))
    if signal_power == 0:
        return None
    error_power = float(np.mean((changed - original) ** 2))
    if error_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / error_power))


def clipping_fraction(audio: np.ndarray) -> float:
    """Fraction of output samples at or beyond full scale (abs(sample) >= 1)."""
    return float(np.mean(np.abs(_mono_finite(audio, "audio")) >= 1.0))


def stoi_score(
    reference: np.ndarray,
    degraded: np.ndarray,
    sample_rate: int,
    sample_aligned: bool = True,
) -> float | None:
    """Optional standard STOI; None for invalid alignment or insufficient speech.

    An absent pystoi installation also produces None. Callers must label missing
    values as inapplicable/unavailable rather than replacing them with zero.
    """
    original = _mono_finite(reference, "reference")
    changed = _mono_finite(degraded, "degraded")
    if not sample_aligned or original.shape != changed.shape or not np.any(original):
        return None
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer.")
    try:
        from pystoi import stoi
    except ImportError:
        return None
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        value = float(stoi(original, changed, sample_rate, extended=False))
    if emitted or not np.isfinite(value):
        return None
    return value


def clip_bootstrap_mean(
    values: Iterable[float],
    seed: int = 20260905,
    n_resamples: int = 2000,
    confidence: float = 0.95,
) -> dict[str, float | int | None]:
    """Percentile bootstrap of the mean, resampling whole clip observations.

    Each supplied value must represent one clip (e.g., its 16-bit accuracy).
    Never pass individual message bits or separate rows for the same clip.
    Non-finite input raises: missing observations must be explicitly filtered
    and counted by the caller before aggregation.
    """
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1000:
        raise ValueError("At least 1,000 clip bootstrap resamples are required.")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one.")
    observations = np.asarray(list(values), dtype=np.float64)
    if observations.ndim != 1 or not np.isfinite(observations).all():
        raise ValueError("Each bootstrap value must be one finite clip-level scalar.")
    count = observations.size
    result: dict[str, float | int | None] = {
        "mean": None,
        "ci_low": None,
        "ci_high": None,
        "n_clips": int(count),
        "n_resamples": int(n_resamples),
    }
    if count == 0:
        return result
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, count, size=(n_resamples, count))
    means = observations[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    result.update({"mean": float(observations.mean()), "ci_low": float(low), "ci_high": float(high)})
    return result
