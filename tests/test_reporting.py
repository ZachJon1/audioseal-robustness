"""Reporting tests use small synthetic clip rows, never model inference."""
import json

import numpy as np
import pandas as pd
import pytest

from audio_wm_eval.reporting import (
    REQUIRED_COLUMNS, aggregate_results, dataset_composition, generate_report,
    load_raw, paired_changes, wilson_interval,
)


def fixture_rows():
    rows = []
    for condition in ("clean", "noise_20"):
        for clip in range(4):
            for branch in ("positive", "negative"):
                positive = branch == "positive"
                errors = 0 if condition == "clean" else clip
                row = {key: "" for key in REQUIRED_COLUMNS}
                row.update(run_id="synthetic", git_commit="abc", clip_id=f"clip_{clip}",
                           speaker_id=f"speaker_{clip // 2}", source_split="test-clean",
                           duration_seconds=3 + clip, seed=20260905, branch=branch,
                           is_watermarked=int(positive), condition=condition,
                           attack_family="clean" if condition == "clean" else "noise",
                           attack_setting="{}", detector_threshold=.5, frame_threshold=.5,
                           detection_score=.9 if positive else .1, mean_frame_probability=.8 if positive else .2,
                           detected=int(positive and (condition == "clean" or clip < 2)),
                           expected_message="0000000000000001", recovered_message="0000000000000001" if positive else "",
                           output_duration_seconds=3 + clip, num_samples=16000 * (3 + clip),
                           output_peak=.5, clipping_fraction=0, prelimit_clipping_fraction=0,
                           embed_runtime_ms=20 if positive else 0, attack_runtime_ms=3,
                           detect_runtime_ms=10, status="ok", attack_metadata="{}",
                           output_sha256="a" * 64, quality_applicability="aligned")
                if positive:
                    row.update(bit_errors=errors, bit_accuracy=1 - errors / 16,
                               bit_error_rate=errors / 16, exact_message_recovery=int(errors == 0),
                               watermark_snr_db=30, watermark_stoi=.99,
                               quality_snr_db=30 if condition == "clean" else 20, stoi=.95)
                else:
                    row.update(quality_snr_db=float("inf") if condition == "clean" else 20, stoi=1)
                rows.append(row)
    return rows


def write_raw(tmp_path, rows=None):
    path = tmp_path / "raw.csv"
    pd.DataFrame(fixture_rows() if rows is None else rows).to_csv(path, index=False)
    return path


def test_raw_keeps_payload_leading_zeroes_and_counts_clips(tmp_path):
    raw = load_raw(write_raw(tmp_path))
    assert raw.expected_message.iloc[0] == "0000000000000001"
    composition = dataset_composition(raw)
    assert composition.n_clips.sum() == 4
    assert composition.duration_total_seconds.sum() == 18
    assert len(composition) == 2


def test_summary_clip_rates_and_paired_bootstrap(tmp_path):
    raw = load_raw(write_raw(tmp_path))
    summary = aggregate_results(raw, resamples=1000).set_index("condition")
    assert summary.loc["clean", "tpr_mean"] == 1
    assert summary.loc["noise_20", "tpr_mean"] == .5
    assert summary.loc["noise_20", "ber_mean"] == 1.5 / 16
    assert summary.loc["noise_20", "exact_recovery_mean"] == .25
    assert summary.loc["noise_20", "ber_n"] == 4  # Four clips, not 64 bits.
    assert summary.loc["clean", "fpr_ci_high"] == 0
    assert summary.loc["clean", "fpr_wilson_ci_high"] > 0
    assert np.isinf(summary.loc["clean", "negative_quality_snr_db_mean"])
    assert summary.loc["clean", "negative_quality_snr_db_n_infinite"] == 4
    assert pd.isna(summary.loc["clean", "negative_quality_snr_db_ci_low"])
    changes = paired_changes(raw, resamples=1000)
    item = changes.loc[(changes.condition == "noise_20") & (changes.metric == "tpr")].iloc[0]
    assert item.attack_minus_clean_mean == -.5
    assert item.n_pairs_attempted == 4
    assert item.attack_minus_clean_n == 4
    assert item.attack_minus_clean_ci_high <= 0


def test_failures_have_explicit_denominators_and_bounds(tmp_path):
    rows = fixture_rows()
    for row in rows:
        if row["condition"] == "noise_20" and row["clip_id"] == "clip_0":
            row.update(status="failed", error_message="synthetic failure", detected="")
    summary = aggregate_results(load_raw(write_raw(tmp_path, rows)), resamples=1000).set_index("condition")
    noise = summary.loc["noise_20"]
    assert noise.n_failed == 2
    assert noise.n_positive_attempted == noise.n_negative_attempted == 4
    assert noise.tpr_n == noise.fpr_n == 3
    assert noise.tpr_n_missing == noise.fpr_n_missing == 1
    assert noise.tpr_mean == 1 / 3
    assert noise.tpr_failed_as_missed == .25
    assert noise.fpr_failure_lower_bound == 0
    assert noise.fpr_failure_upper_bound == .25


@pytest.mark.parametrize("mutation", ["missing_column", "missing_row", "duplicate", "negative_message"])
def test_raw_completeness_rejections(tmp_path, mutation):
    rows = fixture_rows()
    if mutation == "missing_column":
        for row in rows:
            del row["status"]
    elif mutation == "missing_row":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    else:
        rows[1]["bit_accuracy"] = .5
    with pytest.raises(ValueError):
        load_raw(write_raw(tmp_path, rows))


def test_wilson_zero_is_not_zero_upper_bound():
    low, high = wilson_interval(0, 24)
    assert low == pytest.approx(0)
    assert high == pytest.approx(.13797620467498)


def test_aggregation_reproducible_and_resample_floor(tmp_path):
    raw = load_raw(write_raw(tmp_path))
    pd.testing.assert_frame_equal(aggregate_results(raw, 1000), aggregate_results(raw, 1000))
    with pytest.raises(ValueError, match="1,000"):
        aggregate_results(raw, 999)


def test_generate_report_retains_failures_and_refuses_overwrite(tmp_path):
    raw_path = write_raw(tmp_path)
    output = tmp_path / "outputs"
    reports = tmp_path / "reports"
    metadata = generate_report(raw_path, output, reports, resamples=1000)
    assert metadata["n_raw_rows"] == 16
    assert metadata["n_clips"] == 4
    assert metadata == json.loads((output / "aggregation_metadata.json").read_text())
    assert pd.read_csv(output / "failures.csv").empty
    assert len(list((output / "figures").glob("*.png"))) == 4
    assert "not evidence" in (reports / "technical_report.md").read_text()
    assert "No training or fine-tuning" in (reports / "panel_summary.md").read_text()
    assert len((reports / "panel_summary.md").read_text().split()) < 550
    with pytest.raises(FileExistsError):
        generate_report(raw_path, output, reports, resamples=1000)
