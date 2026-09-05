"""Tests exercise real codecs/DSP and deterministic branch pairing."""

import numpy as np
import pytest

from audio_wm_eval.attacks import apply_attack
from audio_wm_eval.metrics import aligned_snr


SEED = 20260905
SAMPLE_RATE = 16000
CONDITIONS = [
    {"id": "clean", "family": "clean"},
    *[{"id": f"mp3_{bitrate}", "family": "mp3", "bitrate_kbps": bitrate} for bitrate in (128, 64)],
    *[{"id": f"noise_{snr}", "family": "noise", "snr_db": snr} for snr in (30, 20)],
    *[{"id": f"resample_{rate}", "family": "resample", "intermediate_sr": rate} for rate in (12000, 8000)],
    *[{"id": f"pitch_{steps}", "family": "pitch", "semitones": steps} for steps in (-2, 2)],
    *[{"id": f"stretch_{rate}", "family": "stretch", "rate": rate} for rate in (0.9, 1.1)],
    *[{"id": f"crop_{fraction}", "family": "crop", "fraction": fraction} for fraction in (0.1, 0.25)],
]


@pytest.fixture
def audio():
    time = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 3.7 * time) ** 2
    return (envelope * (0.25 * np.sin(2 * np.pi * 227 * time) + 0.08 * np.sin(2 * np.pi * 1501 * time))).astype(np.float32)


@pytest.mark.parametrize("condition", CONDITIONS, ids=lambda condition: condition["id"])
def test_all_attacks_shape_range_metadata_and_reproducibility(audio, condition):
    original = audio.copy()
    first, metadata = apply_attack(audio, SAMPLE_RATE, condition, SEED)
    second, repeated_metadata = apply_attack(audio, SAMPLE_RATE, condition, SEED)
    assert first.ndim == 1
    assert first.dtype == np.float32
    assert first.size > 0
    assert np.isfinite(first).all()
    assert np.max(np.abs(first)) <= 1
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(audio, original)
    assert metadata == repeated_metadata
    assert metadata["sample_aligned"] == (condition["family"] in {"clean", "noise", "resample"})
    assert metadata["id"] == condition["id"]
    assert metadata["input_samples"] == audio.size
    assert metadata["output_samples"] == first.size
    assert metadata["duration_seconds"] == first.size / SAMPLE_RATE
    assert 0 <= metadata["clipping_fraction"] <= 1
    if condition["family"] in {"clean", "noise", "resample", "pitch"}:
        assert first.size == audio.size
    elif condition["family"] == "stretch":
        assert first.size == round(audio.size / condition["rate"])
    elif condition["family"] == "crop":
        assert first.size == round(audio.size * (1 - condition["fraction"]))
        np.testing.assert_array_equal(first, audio[metadata["crop_start_sample"] : metadata["crop_end_sample"]])
    if condition["family"] == "clean":
        np.testing.assert_array_equal(first, audio)


@pytest.mark.parametrize("snr_db", [20, 30])
def test_noise_meets_target_snr(audio, snr_db):
    output, metadata = apply_attack(audio, SAMPLE_RATE, {"family": "noise", "snr_db": snr_db}, SEED)
    assert metadata["clipping_fraction"] == 0
    assert metadata["noise_snr_db_pre_clipping"] == pytest.approx(snr_db, abs=1e-10)
    assert metadata["noise_snr_db"] == pytest.approx(snr_db, abs=1e-4)
    assert aligned_snr(audio, output) == pytest.approx(snr_db, abs=1e-4)


def test_noise_reports_limiter_and_achieved_snr():
    original = np.full(SAMPLE_RATE, 0.99, dtype=np.float32)
    output, metadata = apply_attack(original, SAMPLE_RATE, {"family": "noise", "snr_db": 20}, SEED)
    assert metadata["clipping_fraction"] > 0
    assert metadata["pre_limiter_peak"] > 1
    assert output.max() == 1
    assert metadata["noise_snr_db_pre_clipping"] == pytest.approx(20)
    assert metadata["noise_snr_db"] == pytest.approx(aligned_snr(original, output))
    assert abs(metadata["noise_snr_db"] - 20) > 0.1


def test_noise_pairing_reuses_direction_at_branch_specific_scale(audio):
    positive = (audio * 1.01).astype(np.float32)
    condition = {"family": "noise", "snr_db": 20}
    negative_output, _ = apply_attack(audio, SAMPLE_RATE, condition, SEED)
    positive_output, _ = apply_attack(positive, SAMPLE_RATE, condition, SEED)
    negative_noise = negative_output.astype(np.float64) - audio
    positive_noise = positive_output.astype(np.float64) - positive
    np.testing.assert_allclose(
        negative_noise / np.linalg.norm(negative_noise),
        positive_noise / np.linalg.norm(positive_noise),
        atol=2e-8,
    )


@pytest.mark.parametrize("condition", [{"family": "noise", "snr_db": 20}, {"family": "crop", "fraction": 0.25}])
def test_random_attacks_change_with_seed(audio, condition):
    first, _ = apply_attack(audio, SAMPLE_RATE, condition, SEED)
    second, _ = apply_attack(audio, SAMPLE_RATE, condition, SEED + 1)
    assert not np.array_equal(first, second)


def test_matched_crop_uses_same_coordinates(audio):
    condition = {"family": "crop", "fraction": 0.25}
    _, first = apply_attack(audio, SAMPLE_RATE, condition, SEED)
    _, second = apply_attack(audio * np.float32(0.9), SAMPLE_RATE, condition, SEED)
    assert first["crop_start_sample"] == second["crop_start_sample"]
    assert first["crop_end_sample"] == second["crop_end_sample"]


@pytest.mark.parametrize("bad_audio", [
    np.zeros((2, 160), dtype=np.float32),
    np.array([], dtype=np.float32),
    np.array([0, 1], dtype=np.int16),
    np.array([np.nan], dtype=np.float32),
    np.array([np.inf], dtype=np.float32),
    np.array([1.01], dtype=np.float32),
])
def test_rejects_invalid_audio(bad_audio):
    with pytest.raises(ValueError):
        apply_attack(bad_audio, SAMPLE_RATE, {"family": "clean"}, SEED)


@pytest.mark.parametrize("condition", [
    {"family": "unknown"},
    {"family": "noise", "snr_db": float("nan")},
    {"family": "resample", "intermediate_sr": 0},
    {"family": "resample", "intermediate_sr": 32000},
    {"family": "pitch", "semitones": float("inf")},
    {"family": "stretch", "rate": 0},
    {"family": "crop", "fraction": 1},
    {"family": "crop", "fraction": -0.1},
    {"family": "mp3", "bitrate_kbps": 1},
])
def test_rejects_invalid_condition(audio, condition):
    with pytest.raises(ValueError):
        apply_attack(audio, SAMPLE_RATE, condition, SEED)


def test_silent_noise_snr_is_explicitly_undefined():
    with pytest.raises(ValueError, match="undefined for silent"):
        apply_attack(np.zeros(1600, dtype=np.float32), SAMPLE_RATE, {"family": "noise", "snr_db": 20}, SEED)


def test_codec_errors_are_visible(audio):
    with pytest.raises(FileNotFoundError):
        apply_attack(audio, SAMPLE_RATE, {"family": "mp3", "bitrate_kbps": 64}, SEED, ffmpeg_path="/nonexistent/audioseal-ffmpeg")
