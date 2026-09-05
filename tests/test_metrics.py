import numpy as np
import pytest

from audio_wm_eval.metrics import (
    aligned_snr,
    clip_bootstrap_mean,
    clipping_fraction,
    compare_messages,
    stoi_score,
)


def test_message_comparison_counts_all_16_bits():
    expected = "1010101010101010"
    identical = compare_messages(expected, expected)
    assert identical == {"bit_errors": 0, "bit_accuracy": 1, "bit_error_rate": 0, "exact_message_recovery": True}
    three_errors = compare_messages(expected, "0100101010101010")
    assert three_errors == {"bit_errors": 3, "bit_accuracy": 13 / 16, "bit_error_rate": 3 / 16, "exact_message_recovery": False}
    opposite = compare_messages("0" * 16, "1" * 16)
    assert opposite["bit_errors"] == 16
    assert opposite["bit_accuracy"] == 0
    assert opposite["bit_error_rate"] == 1


@pytest.mark.parametrize("bad_message", ["", "0" * 15, "0" * 17, "0" * 15 + "2", "0" * 15 + " ", [0] * 16, None])
def test_invalid_messages_are_not_silently_scored(bad_message):
    with pytest.raises(ValueError):
        compare_messages("0" * 16, bad_message)
    with pytest.raises(ValueError):
        compare_messages(bad_message, "0" * 16)


def test_aligned_snr_known_value():
    reference = np.ones(1600) * 0.5
    degraded = reference + 0.05
    assert aligned_snr(reference, degraded) == pytest.approx(20)
    assert aligned_snr(reference, reference) == float("inf")


def test_incompatible_alignment_silence_and_length_have_no_snr():
    reference = np.ones(1600)
    assert aligned_snr(reference, reference, sample_aligned=False) is None
    assert aligned_snr(reference, reference[:-1]) is None
    assert aligned_snr(np.zeros(1600), reference) is None
    assert aligned_snr(np.zeros(1600), np.zeros(1600)) is None


@pytest.mark.parametrize("invalid", [np.array([]), np.array([np.nan]), np.zeros((2, 32))])
def test_snr_and_clipping_reject_invalid_audio(invalid):
    with pytest.raises(ValueError):
        aligned_snr(invalid, np.ones(32))
    with pytest.raises(ValueError):
        clipping_fraction(invalid)


def test_clipping_fraction_counts_both_polarities_at_full_scale():
    assert clipping_fraction(np.array([-1, -0.99, 0, 0.99, 1])) == 2 / 5
    assert clipping_fraction(np.array([-1.1, 1.1])) == 1


def test_clip_bootstrap_reproducible_and_uses_per_clip_values():
    # Four clips with correlated within-clip bits: the bootstrap gets four
    # whole-message accuracies, never 64 separate correctness observations.
    per_clip = [0, 0, 1, 1]
    first = clip_bootstrap_mean(per_clip, seed=20260905, n_resamples=1000)
    second = clip_bootstrap_mean(per_clip, seed=20260905, n_resamples=1000)
    assert first == second
    assert first["n_clips"] == 4
    assert first["mean"] == 0.5
    assert first["ci_low"] == 0
    assert first["ci_high"] == 1
    assert first["n_resamples"] == 1000


def test_bootstrap_all_zero_and_empty_values():
    zero = clip_bootstrap_mean([0] * 24)
    assert zero["mean"] == zero["ci_low"] == zero["ci_high"] == 0
    empty = clip_bootstrap_mean([])
    assert empty["n_clips"] == 0
    assert empty["mean"] is None
    assert empty["ci_low"] is None
    assert empty["ci_high"] is None


@pytest.mark.parametrize("values", [[0, np.nan], [np.inf], [[0, 1], [1, 0]]])
def test_bootstrap_does_not_hide_nonfinite_or_bit_matrix_input(values):
    with pytest.raises(ValueError):
        clip_bootstrap_mean(values)


def test_bootstrap_requires_1000_or_more_resamples():
    with pytest.raises(ValueError, match="1,000"):
        clip_bootstrap_mean([0, 1], n_resamples=999)


def test_stoi_requires_alignment_and_speech():
    speech = np.sin(np.arange(16000) / 20)
    assert stoi_score(speech, speech[:-1], 16000) is None
    assert stoi_score(speech, speech, 16000, sample_aligned=False) is None
    assert stoi_score(np.zeros(16000), speech, 16000) is None


def test_stoi_identical_speech_like_signal():
    pytest.importorskip("pystoi")
    generator = np.random.default_rng(20260905)
    time = np.arange(32000) / 16000
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 4 * time)) ** 2
    speech_like = envelope * generator.standard_normal(time.size) * 0.1
    assert stoi_score(speech_like, speech_like, 16000) == pytest.approx(1, abs=1e-6)
