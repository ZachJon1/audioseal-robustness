"""Raw-result-only aggregation, figures, and cautious research reports.

Inference failures remain in the input grid and every summary denominator is
explicit. Bootstrap units are clips, never message bits or repeated conditions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import clip_bootstrap_mean

SEED = 20260905
REQUIRED_COLUMNS = {
    "run_id", "git_commit", "clip_id", "speaker_id", "source_split",
    "duration_seconds", "seed", "branch", "is_watermarked", "condition",
    "attack_family", "attack_setting", "detector_threshold", "frame_threshold",
    "detection_score", "mean_frame_probability", "detected", "expected_message",
    "recovered_message", "bit_errors", "bit_accuracy", "bit_error_rate",
    "exact_message_recovery", "watermark_snr_db", "watermark_stoi", "stoi", "pesq",
    "quality_snr_db", "quality_applicability", "output_duration_seconds",
    "num_samples", "output_peak", "clipping_fraction", "prelimit_clipping_fraction",
    "embed_runtime_ms", "attack_runtime_ms", "detect_runtime_ms", "status",
    "error_message", "attack_metadata", "output_sha256",
}
NUMERIC_COLUMNS = REQUIRED_COLUMNS - {
    "run_id", "git_commit", "clip_id", "speaker_id", "source_split", "branch",
    "condition", "attack_family", "attack_setting", "expected_message",
    "recovered_message", "quality_applicability", "status", "error_message",
    "attack_metadata", "output_sha256",
}
LABELS = {
    "clean": "Clean", "mp3_128": "MP3 128 kbps", "mp3_64": "MP3 64 kbps",
    "noise_30": "Noise 30 dB", "noise_20": "Noise 20 dB",
    "resample_12000": "Resample 12 kHz", "resample_8000": "Resample 8 kHz",
    "pitch_minus2": "Pitch −2 st", "pitch_plus2": "Pitch +2 st",
    "stretch_09": "Stretch 0.9×", "stretch_11": "Stretch 1.1×",
    "crop_10": "Crop 10%", "crop_25": "Crop 25%",
}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Preserve message leading zeroes and reject incomplete experimental grids."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required result columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Raw results are empty.")
    for name in NUMERIC_COLUMNS:
        values = frame[name].replace({"True": "1", "False": "0", "true": "1", "false": "0"})
        cleaned = values.mask(values.isin(["", "NA", "N/A", "None", "nan"]))
        frame[name] = pd.to_numeric(cleaned, errors="raise")
    if set(frame["status"]) - {"ok", "failed"}:
        raise ValueError("Result status must be 'ok' or 'failed'.")
    if set(frame["branch"]) != {"positive", "negative"}:
        raise ValueError("Both positive and negative branches are required.")
    keys = ["clip_id", "condition", "branch"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate clip-condition-branch result rows.")
    clips = set(frame.clip_id)
    for condition, rows in frame.groupby("condition", sort=False):
        for branch in ("positive", "negative"):
            if set(rows.loc[rows.branch == branch, "clip_id"]) != clips:
                raise ValueError(f"Incomplete matched result grid: {condition}/{branch}.")
    if frame.run_id.nunique() != 1 or set(frame.seed) != {SEED}:
        raise ValueError("Expected exactly one run and the prescribed seed 20260905.")
    for name in ("detector_threshold", "frame_threshold"):
        if frame[name].isna().any() or frame[name].nunique() != 1:
            raise ValueError(f"Expected one prespecified {name} across all rows.")
    if not (frame.is_watermarked == (frame.branch == "positive").astype(int)).all():
        raise ValueError("Branch/is_watermarked mismatch.")
    ok = frame.status == "ok"
    if frame.loc[ok, "detected"].isna().any() or not frame.loc[ok, "detected"].isin([0, 1]).all():
        raise ValueError("Successful rows require a binary detected value.")
    for name in ("detection_score", "mean_frame_probability", "bit_accuracy", "bit_error_rate", "clipping_fraction"):
        present = frame.loc[ok, name].dropna()
        if not present.between(0, 1).all():
            raise ValueError(f"{name} must lie in [0, 1].")
    negative = frame.branch == "negative"
    for name in ("bit_errors", "bit_accuracy", "bit_error_rate", "exact_message_recovery"):
        if frame.loc[negative, name].notna().any():
            raise ValueError(f"Unwatermarked message metric must be inapplicable: {name}.")
    for name in ("speaker_id", "source_split", "duration_seconds", "expected_message"):
        if frame.groupby("clip_id")[name].nunique().gt(1).any():
            raise ValueError(f"Inconsistent clip metadata: {name}.")
    return frame


def _summary(values: pd.Series, prefix: str, seed: int, resamples: int) -> dict:
    values = pd.to_numeric(values, errors="raise")
    present = values.dropna().to_numpy(dtype=float)
    finite = present[np.isfinite(present)]
    stats = clip_bootstrap_mean(finite, seed=seed, n_resamples=resamples)
    result = {
        f"{prefix}_mean": stats["mean"],
        f"{prefix}_ci_low": stats["ci_low"],
        f"{prefix}_ci_high": stats["ci_high"],
        f"{prefix}_n": int(len(present)),
        f"{prefix}_n_missing": int(values.isna().sum()),
        f"{prefix}_n_finite": int(len(finite)),
        f"{prefix}_n_infinite": int(np.isinf(present).sum()),
    }
    if np.isinf(present).any():
        # An arithmetic mean with an infinite SNR is infinite; never quietly
        # substitute the mean of the remaining finite clips.
        result[f"{prefix}_mean"] = float(np.mean(present))
        result[f"{prefix}_ci_low"] = None
        result[f"{prefix}_ci_high"] = None
    return result


def wilson_interval(successes: int, count: int) -> tuple[float, float]:
    """Two-sided 95% Wilson interval; supplemental only, assuming independent clips."""
    if count == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / count
    divisor = 1 + z * z / count
    center = (p + z * z / (2 * count)) / divisor
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / divisor
    return max(0.0, center - half), min(1.0, center + half)


def aggregate_results(raw: pd.DataFrame, resamples: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """One condition per row, with metric-specific measured-clip denominators."""
    if resamples < 1000:
        raise ValueError("At least 1,000 bootstrap resamples are required.")
    result = []
    for condition, rows in raw.groupby("condition", sort=False):
        pos_all = rows.loc[rows.branch == "positive"]
        neg_all = rows.loc[rows.branch == "negative"]
        pos = pos_all.loc[pos_all.status == "ok"]
        neg = neg_all.loc[neg_all.status == "ok"]
        item = {
            "condition": condition, "attack_family": rows.attack_family.iloc[0],
            "n_clips": rows.clip_id.nunique(), "n_speakers": rows.speaker_id.nunique(),
            "n_positive_attempted": len(pos_all), "n_negative_attempted": len(neg_all),
            "n_positive_ok": len(pos), "n_negative_ok": len(neg),
            "n_positive_failed": len(pos_all) - len(pos),
            "n_negative_failed": len(neg_all) - len(neg),
            "n_failed": int((rows.status == "failed").sum()),
            "detector_threshold": rows.detector_threshold.iloc[0],
            "frame_threshold": rows.frame_threshold.iloc[0], "bootstrap_resamples": resamples,
            "bootstrap_seed": seed,
        }
        # Keep failed attempts as missing entries for denominator/missing counts.
        pos_masked = pos_all.copy()
        neg_masked = neg_all.copy()
        for branch_rows in (pos_masked, neg_masked):
            branch_rows.loc[branch_rows.status != "ok", list(NUMERIC_COLUMNS - {"seed", "is_watermarked"})] = np.nan
        for metric, branch_rows, column in (
            ("tpr", pos_masked, "detected"), ("fpr", neg_masked, "detected"),
            ("ber", pos_masked, "bit_error_rate"), ("bit_accuracy", pos_masked, "bit_accuracy"),
            ("exact_recovery", pos_masked, "exact_message_recovery"),
        ):
            item.update(_summary(branch_rows[column], metric, seed, resamples))
        item.update(_summary(1 - pos_masked.detected, "fnr", seed, resamples))
        item["tpr_failed_as_missed"] = float(pos.detected.sum() / len(pos_all))
        item["fpr_failure_lower_bound"] = float(neg.detected.sum() / len(neg_all))
        item["fpr_failure_upper_bound"] = float((neg.detected.sum() + len(neg_all) - len(neg)) / len(neg_all))
        low, high = wilson_interval(int(neg.detected.sum()), len(neg))
        item.update(fpr_wilson_ci_low=low, fpr_wilson_ci_high=high)
        for branch, branch_rows in (("positive", pos_masked), ("negative", neg_masked)):
            for column in (
                "detection_score", "mean_frame_probability", "quality_snr_db", "stoi", "pesq",
                "output_duration_seconds", "output_peak", "clipping_fraction", "prelimit_clipping_fraction",
                "embed_runtime_ms", "attack_runtime_ms", "detect_runtime_ms",
            ):
                item.update(_summary(branch_rows[column], f"{branch}_{column}", seed, resamples))
        for column in ("watermark_snr_db", "watermark_stoi"):
            item.update(_summary(pos_masked[column], column, seed, resamples))
        item["positive_quality_applicability"] = json.dumps(pos_all.quality_applicability.value_counts().to_dict(), sort_keys=True)
        item["negative_quality_applicability"] = json.dumps(neg_all.quality_applicability.value_counts().to_dict(), sort_keys=True)
        result.append(item)
    return pd.DataFrame(result)


def paired_changes(raw: pd.DataFrame, resamples: int = 2000, seed: int = SEED) -> pd.DataFrame:
    """Paired attack-minus-clean differences, resampling the same clip pair."""
    result = []
    for condition in raw.condition.unique():
        for branch, metric, column in (
            ("positive", "tpr", "detected"), ("negative", "fpr", "detected"),
            ("positive", "ber", "bit_error_rate"), ("positive", "exact_recovery", "exact_message_recovery"),
        ):
            base = raw.loc[(raw.condition == "clean") & (raw.branch == branch), ["clip_id", "status", column]]
            attacked = raw.loc[(raw.condition == condition) & (raw.branch == branch), ["clip_id", "status", column]]
            paired = attacked.merge(base, on="clip_id", suffixes=("_attack", "_clean"), validate="one_to_one")
            valid = ((paired.status_attack == "ok") & (paired.status_clean == "ok")
                     & paired[f"{column}_attack"].notna() & paired[f"{column}_clean"].notna())
            differences = paired.loc[valid, f"{column}_attack"] - paired.loc[valid, f"{column}_clean"]
            item = {"condition": condition, "branch": branch, "metric": metric,
                    "n_pairs_attempted": len(paired), "n_pairs_missing": int((~valid).sum())}
            item.update(_summary(differences, "attack_minus_clean", seed, resamples))
            result.append(item)
    return pd.DataFrame(result)


def dataset_composition(raw: pd.DataFrame) -> pd.DataFrame:
    clips = raw.drop_duplicates("clip_id")
    rows = []
    for (split, speaker), group in clips.groupby(["source_split", "speaker_id"], sort=True):
        rows.append({"source_split": split, "speaker_id": speaker, "n_clips": len(group),
                     "duration_total_seconds": group.duration_seconds.sum(),
                     "duration_mean_seconds": group.duration_seconds.mean(),
                     "duration_min_seconds": group.duration_seconds.min(),
                     "duration_max_seconds": group.duration_seconds.max(),
                     "clip_ids": ";".join(sorted(group.clip_id))})
    return pd.DataFrame(rows)


def _figures(raw: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 140, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    labels = [LABELS.get(c, c) for c in summary.condition]
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for metric, offset, color, label in (("tpr", -.1, "#126e82", "TPR"), ("fpr", .1, "#b13e53", "FPR")):
        mean = summary[f"{metric}_mean"].to_numpy(dtype=float)
        low = summary[f"{metric}_ci_low"].to_numpy(dtype=float)
        high = summary[f"{metric}_ci_high"].to_numpy(dtype=float)
        ax.errorbar(x + offset, mean, yerr=[np.maximum(0, mean - low), np.maximum(0, high - mean)],
                    fmt="o", capsize=3, color=color, label=label)
    ax.set(xticks=x, xticklabels=labels, ylim=(-.05, 1.05), ylabel="Measured clip proportion",
           title="AudioSeal detection: clip-bootstrap 95% intervals")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output / "detection_rates.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.7), sharex=True)
    for ax, metric, label in zip(axes, ("ber", "exact_recovery"), ("Bit error rate", "Exact 16-bit recovery")):
        mean = summary[f"{metric}_mean"].to_numpy(dtype=float)
        low = summary[f"{metric}_ci_low"].to_numpy(dtype=float)
        high = summary[f"{metric}_ci_high"].to_numpy(dtype=float)
        ax.errorbar(x, mean, yerr=[np.maximum(0, mean - low), np.maximum(0, high - mean)], fmt="o", color="#126e82", capsize=3)
        ax.set(xticks=x, xticklabels=labels, ylabel=label, ylim=(-.05, 1.05), title=label + " (positive clips)")
        ax.tick_params(axis="x", rotation=70)
    fig.suptitle("Recovery scored on all successfully processed positive clips, including missed detections")
    fig.tight_layout()
    fig.savefig(output / "message_recovery.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.7))
    for branch, offset, color in (("positive", -.17, "#126e82"), ("negative", .17, "#b13e53")):
        values = [raw.loc[(raw.condition == c) & (raw.branch == branch) & (raw.status == "ok"), "detection_score"].dropna().to_numpy()
                  for c in summary.condition]
        # Box plots show the clip distribution without treating rows in different
        # conditions as independent observations.
        nonempty = [i for i, values_i in enumerate(values) if len(values_i)]
        if nonempty:
            bp = ax.boxplot([values[i] for i in nonempty], positions=np.array(nonempty) + offset, widths=.27, patch_artist=True,
                            manage_ticks=False, showfliers=True)
            for box in bp["boxes"]:
                box.set(facecolor=color, alpha=.55)
        ax.plot([], [], color=color, linewidth=6, alpha=.6, label=branch.capitalize())
    thresholds = raw.detector_threshold.dropna().unique()
    if len(thresholds) == 1:
        ax.axhline(thresholds[0], color="gray", linestyle="--", label="Prespecified clip threshold")
    ax.set(xticks=x, xticklabels=labels, ylabel="Fraction of frames declared watermarked", ylim=(-.05, 1.05),
           title="Detection-score distributions across clips")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "score_distributions.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    finite = np.isfinite(pd.to_numeric(summary.positive_quality_snr_db_mean, errors="coerce"))
    quality = summary.loc[finite]
    if len(quality):
        ax.scatter(quality.positive_quality_snr_db_mean, quality.tpr_mean, color="#126e82")
        for row in quality.itertuples():
            ax.annotate(LABELS.get(row.condition, row.condition), (row.positive_quality_snr_db_mean, row.tpr_mean),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    else:
        ax.text(.5, .5, "No applicable finite sample-aligned SNR values", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Mean positive-output sample-aligned SNR vs original speech (dB)",
           ylabel="TPR", ylim=(-.05, 1.12), title="Applicable aligned conditions only; signal quality ≠ listening quality")
    fig.tight_layout()
    fig.savefig(output / "quality_vs_detection.png")
    plt.close(fig)


def _pct(value) -> str:
    return "NA" if pd.isna(value) else f"{100 * value:.1f}%"


def _number(value, digits=2) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{value:.{digits}f}" if np.isfinite(value) else ("∞" if value > 0 else "−∞")


def _rate_interval(row, metric) -> str:
    return f"{_pct(row[f'{metric}_mean'])} [{_pct(row[f'{metric}_ci_low'])}, {_pct(row[f'{metric}_ci_high'])}]"


def _table(summary: pd.DataFrame, intervals: bool = True) -> str:
    metrics = [("tpr", "TPR"), ("fpr", "FPR")]
    if intervals:
        metrics += [("fnr", "FNR"), ("bit_accuracy", "Bit accuracy")]
    metrics += [("ber", "BER"), ("exact_recovery", "Exact 16-bit")]
    lines = ["| Condition | " + " | ".join(label for _, label in metrics) + " | Failed rows |",
             "|---|" + "---:|" * (len(metrics) + 1)]
    for _, row in summary.iterrows():
        value = lambda m: _rate_interval(row, m) if intervals else _pct(row[f"{m}_mean"])
        lines.append(f"| {LABELS.get(row.condition, row.condition)} | "
                     + " | ".join(value(metric) for metric, _ in metrics) + f" | {int(row.n_failed)} |")
    return "\n".join(lines)


def _numeric_interval(row, metric: str, digits: int = 2) -> str:
    return (f"{_number(row[f'{metric}_mean'], digits)} "
            f"[{_number(row[f'{metric}_ci_low'], digits)}, {_number(row[f'{metric}_ci_high'], digits)}]")


def _quality_runtime_tables(summary: pd.DataFrame) -> str:
    quality = ["| Condition | Positive SNR (dB) | Negative SNR (dB) | Positive STOI | Positive duration (s) | Positive clipping |",
               "|---|---:|---:|---:|---:|---:|"]
    runtime = ["| Condition | Positive attack (ms) | Negative attack (ms) | Positive detection (ms) | Negative detection (ms) |",
               "|---|---:|---:|---:|---:|"]
    for _, row in summary.iterrows():
        label = LABELS.get(row.condition, row.condition)
        quality.append(f"| {label} | {_numeric_interval(row, 'positive_quality_snr_db')} | "
                       f"{_numeric_interval(row, 'negative_quality_snr_db')} | {_numeric_interval(row, 'positive_stoi', 3)} | "
                       f"{_numeric_interval(row, 'positive_output_duration_seconds', 3)} | "
                       f"{_rate_interval(row, 'positive_clipping_fraction')} |")
        runtime.append(f"| {label} | " + " | ".join(_numeric_interval(row, name) for name in
                       ("positive_attack_runtime_ms", "negative_attack_runtime_ms",
                        "positive_detect_runtime_ms", "negative_detect_runtime_ms")) + " |")
    return "\n".join(quality) + "\n\n" + "\n".join(runtime)


def _optional_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Evidence file must contain a JSON object: {path}")
    return value


def _project_root(raw_path: str | Path) -> Path:
    """Find experiment evidence near the input, never from an unrelated cwd."""
    parent = Path(raw_path).resolve().parent
    for candidate in (parent, *parent.parents):
        if (candidate / "outputs/environment.json").is_file() or (candidate / "configs/experiment.yaml").is_file():
            return candidate
    return parent


def _environment_table(environment: dict, run_manifest: dict) -> str:
    packages = environment.get("packages", {})
    config = run_manifest.get("config", {})
    provenance = environment.get("model_provenance", {})
    cpu_text = environment.get("cpu", {}).get("stdout", "")
    cpu_model = next((line.split(":", 1)[1].strip() for line in cpu_text.splitlines()
                      if line.strip().startswith("Model name:")), "not recorded")
    ffmpeg_text = environment.get("ffmpeg", {}).get("stdout", "")
    ffmpeg_version = ffmpeg_text.splitlines()[0] if ffmpeg_text else "not recorded"
    settings = [
        ("Python", environment.get("python", "not recorded")),
        ("Host / CPU", f"{environment.get('platform', 'not recorded')}; {cpu_model}; {environment.get('cpu_count', 'NA')} logical CPUs"),
        ("Execution", f"device={config.get('device', environment.get('device', 'NA'))}; "
         f"torch_threads={config.get('torch_threads', environment.get('torch_threads', 'NA'))}; "
         f"CUDA available={environment.get('cuda_available', 'NA')}; "
         f"deterministic_algorithms={environment.get('deterministic_algorithms', 'NA')}; "
         f"torch_compile_disabled={environment.get('torch_compile_disabled', 'NA')}"),
        ("Core software", "; ".join(f"{name}=={packages.get(name, 'not recorded')}" for name in
                                  ("audioseal", "torch", "torchaudio", "numpy", "scipy", "librosa", "soundfile", "pystoi"))),
        ("FFmpeg", ffmpeg_version),
        ("Generator / detector", f"{config.get('generator', 'not recorded')} / {config.get('detector', 'not recorded')}"),
        ("Embedding / payload", f"strength={config.get('watermark_strength', 'NA')}; 16 bits; seed={config.get('seed', environment.get('seed', 'NA'))}"),
        ("Checkpoint revision", provenance.get("model_revision", "not recorded")),
        ("Generator SHA-256", provenance.get("generator", {}).get("sha256", "not recorded")),
        ("Detector SHA-256", provenance.get("detector", {}).get("sha256", "not recorded")),
        ("Dependency-lock SHA-256", environment.get("lock_sha256", "not recorded")),
    ]
    return "\n".join(["| Recorded item | Value |", "|---|---|"] +
                     [f"| {name} | {str(value).replace('|', '/').replace(chr(10), ' ')} |" for name, value in settings])


def _attack_table(raw: pd.DataFrame) -> str:
    mechanics = {
        "clean": "Identity; no post-processing other than the common range policy.",
        "mp3": "FFmpeg libmp3lame encode/decode at 16 kHz; codec delay not independently aligned; SNR/STOI NA.",
        "noise": "One seeded standard-normal draw; scaled separately to each branch's RMS for target SNR before limiting.",
        "resample": "SciPy polyphase down/up sampling, Kaiser window beta=5; output trimmed to input sample count.",
        "pitch": "librosa phase-vocoder pitch shift, 12 bins/octave, soxr_hq; default FFT=2048/hop=512; SNR/STOI NA.",
        "stretch": "librosa phase-vocoder stretch; default FFT=2048/hop=512; output length approximately input/rate; SNR/STOI NA.",
        "crop": "Retain a contiguous round(N*(1-fraction)) window at a seeded random valid start; SNR/STOI NA.",
    }
    lines = ["| Condition | Recorded parameters | Operation / quality applicability |", "|---|---|---|"]
    for condition, rows in raw.groupby("condition", sort=False):
        family = rows.attack_family.iloc[0]
        settings = json.loads(rows.attack_setting.iloc[0])
        parameters = ", ".join(f"{key}={value}" for key, value in sorted(settings.items()) if key not in {"id", "family"}) or "none"
        lines.append(f"| {LABELS.get(condition, condition)} | {parameters} | {mechanics.get(family, 'See raw attack_metadata.')} |")
    return "\n".join(lines)


def _validation_evidence(project: Path, raw_path: Path, reports: Path, environment: dict) -> tuple[str, str]:
    artifacts = [("Official smoke retry", project / "outputs/smoke_test.json"),
                 ("Pilot inspection", project / "outputs/pilot/inspection.json"),
                 ("Raw run validation", raw_path.resolve().parent / "validation.json"),
                 ("Final test suite", project / "outputs/final_validation.json"),
                 ("Reproducibility audit", project / "outputs/reproducibility_audit.json"),
                 ("Claims audit", project / "outputs/claims_audit.json")]
    lines = ["| Validation stage | Recorded result | Artifact |", "|---|---|---|"]
    for label, path in artifacts:
        evidence = _optional_json(path)
        if not evidence:
            lines.append(f"| {label} | not present at generation; consult the final STATUS.md audit record | — |")
            continue
        detail = [f"status={evidence.get('status', 'not specified')}"]
        for key in ("actual_rows", "expected_rows", "failed_rows", "tests_passed", "tests_failed", "matched_rows", "summary"):
            if key in evidence and isinstance(evidence[key], (str, int, float, bool)):
                detail.append(f"{key}={evidence[key]}")
        text = "; ".join(detail).replace("|", "/").replace("\n", " ")
        relative = Path(os.path.relpath(path, reports)).as_posix()
        lines.append(f"| {label} | {text} | [{path.name}]({relative}) |")
    smoke_failures = sorted((project / "outputs").glob("smoke_test_failed*.json"))
    setup = "No earlier smoke-failure artifact was present in the evidence directory when this report was generated."
    if smoke_failures:
        first = _optional_json(smoke_failures[0])
        error = str(first.get("error_message", "unspecified failure"))
        relative = Path(os.path.relpath(smoke_failures[0], reports)).as_posix()
        if "setuptools" in error:
            setup = ("The first official-example smoke attempt failed in PyTorch's optional Inductor compilation "
                     "with `ModuleNotFoundError: No module named 'setuptools'`. "
                     f"The recorded environment includes `setuptools=={environment.get('packages', {}).get('setuptools', 'not recorded')}` "
                     f"and `torch_compile_disabled={environment.get('torch_compile_disabled', 'not recorded')}`. "
                     "The successful retry used the official model's eager inference path with `TORCHDYNAMO_DISABLE=1`; "
                     "the official AudioSeal implementation and checkpoints were not substituted. "
                     f"The original failure and traceback are retained in [{smoke_failures[0].name}]({relative}).")
        else:
            setup = f"An earlier smoke attempt failed: `{error.splitlines()[0]}`. See [{smoke_failures[0].name}]({relative})."
    return setup, "\n".join(lines)


def _write_reports(raw: pd.DataFrame, summary: pd.DataFrame, paired: pd.DataFrame, metadata: dict, reports: Path, output: Path) -> None:
    clips = raw.drop_duplicates("clip_id")
    n = len(clips)
    speakers = clips.speaker_id.nunique()
    failures = int((raw.status == "failed").sum())
    clean_rows = summary.loc[summary.condition == "clean"]
    clean = clean_rows.iloc[0] if len(clean_rows) else None
    report_rel = Path(os.path.relpath(output, reports))
    source_links = (
        "[AudioSeal paper](https://proceedings.mlr.press/v235/san-roman24a.html); "
        "[official implementation](https://github.com/facebookresearch/audioseal); "
        "[robustness framing](https://arxiv.org/abs/2503.19176); "
        "[benchmark design reference](https://doi.org/10.1109/ACCESS.2026.3685903)."
    )
    project = _project_root(metadata["raw_path"])
    environment = _optional_json(project / "outputs/environment.json")
    run_manifest = _optional_json(Path(metadata["raw_path"]).resolve().parent / "run_manifest.json")
    setup_failures, validation_table = _validation_evidence(project, Path(metadata["raw_path"]), reports, environment)
    clean_text = "Clean-condition results unavailable."
    if clean is not None:
        clean_text = (f"Clean TPR was {_rate_interval(clean, 'tpr')}; FPR was {_rate_interval(clean, 'fpr')}. "
                      f"Clean BER was {_rate_interval(clean, 'ber')} and exact 16-bit recovery was {_rate_interval(clean, 'exact_recovery')}.")
    zero_conditions = summary.loc[(summary.fpr_mean == 0) & (summary.fpr_n > 0)]
    false_positive_note = ""
    if len(zero_conditions):
        upper = zero_conditions.fpr_wilson_ci_high.max()
        false_positive_note = (f"Zero false positives occurred in {len(zero_conditions)} condition(s). Their empirical bootstrap intervals collapse to [0, 0]; "
                               f"this is not evidence that the population FPR is zero. The largest supplemental two-sided 95% Wilson upper bound for these conditions is {_pct(upper)}. "
                               "Wilson bounds also assume independent clips and do not repair speaker dependence.")
    attacks = summary.loc[summary.condition != "clean"]
    valid_attacks = attacks.loc[attacks.tpr_mean.notna()]
    if len(valid_attacks):
        worst = valid_attacks.loc[valid_attacks.tpr_mean.idxmin()]
        attack_text = (f"Among tested transformations, the lowest measured TPR was {_pct(worst.tpr_mean)} for {LABELS.get(worst.condition, worst.condition)} "
                       f"(BER {_pct(worst.ber_mean)}, exact recovery {_pct(worst.exact_recovery_mean)}). Ties are visible in the full table.")
    else:
        worst = None
        attack_text = "No successfully measured attacked-condition TPR was available."
    positive_ok = raw.loc[(raw.branch == "positive") & (raw.status == "ok")]
    negative_ok = raw.loc[(raw.branch == "negative") & (raw.status == "ok")]
    # A clip's preattack watermark quality and embedding runtime appear in every
    # attack row. Use one positive observation per clip for the overall narrative.
    once = positive_ok.drop_duplicates("clip_id")
    wm_snr = once.watermark_snr_db.dropna()
    wm_stoi = once.watermark_stoi.dropna()
    max_clip = raw.loc[raw.status == "ok", "clipping_fraction"].max()
    runtimes = (
        f"Embedding median {_number(once.embed_runtime_ms.median())} ms/clip ({len(once)} distinct positive clips); "
        f"detection median {_number(pd.concat([positive_ok, negative_ok]).detect_runtime_ms.median())} ms per successful branch-condition inference. "
        "Per-row embedding/detection timing excludes model loading and the dedicated model warmup. "
        "Attack timing includes codec I/O and first library imports; cache effects and fixed positive-before-negative order can affect timing. "
        "Quality calculations and example-file writes contribute only to total run wall time. These are host-specific measurements, not a cross-hardware benchmark."
    )
    baseline_stats = {}
    for column in ("watermark_snr_db", "watermark_stoi", "embed_runtime_ms"):
        baseline_stats.update(_summary(once[column], column, SEED, metadata["bootstrap_resamples"]))
    negative_aligned = negative_ok.quality_snr_db.notna().sum()
    positive_aligned = positive_ok.quality_snr_db.notna().sum()
    technical = f"""# Preliminary AudioSeal robustness evaluation

This reproducible baseline evaluates one official pretrained AudioSeal generator/detector pair on {n} public speech clips from {speakers} speakers. It is a small, controlled inference experiment, with no training or fine-tuning. It does not establish general audio-watermarking robustness, security, or perceptual transparency.

## Results at a glance

{clean_text} {attack_text} A total of {failures} of {len(raw)} attempted branch-condition rows failed. Rates below use successfully measured clips; attempted counts, missing metric counts, failed-positive-as-missed TPR, and negative-failure FPR bounds appear in the CSV.

## Sources and relation to prior work

The AudioSeal paper and official repository define the evaluated system. The supplied SoK frames broader robustness evaluation. The requested 2026 DeepMark paper's identity was verified through its official repository, whose documentation informed modular attack design, matched controls, and alignment cautions; the 2026 article's full text was unavailable. This project evaluates its own small prespecified subset and does not reproduce those papers' experimental claims. Source-specific access failures and verification notes are in [source_notes.md](source_notes.md).

{source_links}

## Data, model, and experimental design

- Data: {', '.join(sorted(clips.source_split.unique()))}; {n} clips and {speakers} speakers, with observed durations {_number(clips.duration_seconds.min())}–{_number(clips.duration_seconds.max())} seconds and {_number(clips.duration_seconds.sum())} seconds total. Inputs are mono 16 kHz. Selection and expected 16-bit messages are recorded in the manifest; prepared-file hashes allow content checks.
- Model: official `audioseal==0.2.0`, generator `audioseal_wm_16bits`, detector `audioseal_detector_16bits`, official pretrained checkpoints. Exact environment and checkpoint hashes are retained in project outputs. No substitute model, training, or fine-tuning was used.
- Seed: {SEED}. Every clip has a deterministic 16-bit payload. The positive branch receives an additive watermark at strength 1; its matched negative branch receives no watermark. Both receive the same configured transformation and deterministic attack seed for that clip and condition.
- Conditions: clean; MP3 128/64 kbps; Gaussian noise at 30/20 dB target SNR; downsample/upsample through 12/8 kHz; pitch ±2 semitones; librosa time-stretch rates 0.9/1.1 (output duration approximately input duration divided by rate); and removal of 10%/25% of duration by deterministic cropping. Exact mechanics and attack diagnostics are retained in raw `attack_metadata`; settings are in `configs/attacks.yaml`.
- The official detector yields frame probabilities and message probabilities. A frame is positive when its watermark probability exceeds {raw.frame_threshold.iloc[0]:g}. The clip score is the fraction of positive frames. This study prespecifies clip detection as score > {raw.detector_threshold.iloc[0]:g}; this clip decision is not asserted to be an officially calibrated deployment default. Thresholds were not selected using these evaluation outcomes.
- The official example-audio smoke test gated the dataset download. A two-clip pilot covered all conditions before the full run; pilot and smoke artifacts are retained separately.

The following configuration and software values come from saved run/environment evidence, rather than the current shell or unrecorded defaults. The complete dependency list and provenance remain in `outputs/environment.json`.

{_environment_table(environment, run_manifest)}

{_attack_table(raw)}

Every output uses the common range policy: any samples outside [−1, 1] are limited before detection, with the prelimit rate recorded separately. Matched noise branches share the same random realization, scaled to their own RMS. Cropping removes a total fraction by retaining one contiguous window; it does not remove an interior segment and concatenate the remainder. The manifest selection occurs before inference and uses only source eligibility, sorted identities, and the fixed seed.

## Metrics and uncertainty

TPR = detected positive clips / successfully measured positive clips; FNR = 1 − TPR; FPR = detected negative controls / successfully measured negative controls. Failures are explicitly reported and are not silently converted into valid detector outcomes. The summary also provides TPR with failed positives counted as misses and lower/upper possible FPR bounds for failed controls.

Bit accuracy and BER compare the recovered payload with the expected 16-bit payload for every successfully processed positive clip, including undetected ones. Exact recovery requires all 16 bits correct. Negative-control message accuracy is inapplicable because no message was embedded. Each clip contributes one scalar for each recovery metric; individual bits are never treated as independent samples.

All reported bootstrap intervals are 95% percentile intervals from {metadata['bootstrap_resamples']:,} resamples with seed {SEED}, resampling clips within a condition. `paired_differences.csv` resamples per-clip attack-minus-clean differences. The same clips recur across conditions, so conditions are paired and must not be pooled as independent observations. There are only {speakers} speakers; multiple clips per speaker violate a strict independent-clip interpretation. These intervals describe clip-level sampling uncertainty in this small selected set, may understate speaker-level uncertainty, and do not include model, dataset, threshold, or attack-selection uncertainty. No multiple-comparison significance claims are made.

{false_positive_note}

Watermark SNR compares original speech with its watermarked version before attack, once per clip when reported overall. Positive-output quality SNR compares the original speech with the final positive output and therefore includes both watermark and attack distortion; the matched negative comparison captures attack-only distortion. SNR is reported only when equal length and verified sample alignment make it meaningful. Pitch shifting, time stretching, cropping, and any unverified codec alignment are explicitly inapplicable for sample-aligned quality metrics. Identical aligned signals have infinite SNR; such values are counted explicitly, and their bootstrap intervals are left inapplicable. Clipping rate is the fraction of samples with absolute amplitude ≥ 1; prelimit clipping diagnostics are kept separately.

Standard STOI is included only for compatible aligned, nonempty speech when its implementation returns a valid score. PESQ/ViSQOL were not added. Objective sample error and intelligibility estimates do not establish listening quality or watermark inaudibility; no human listening study was completed. Quality SNR was populated for {positive_aligned} successful positive and {negative_aligned} successful negative rows, with per-condition applicability recorded in the CSV.

## Full per-condition results

Each bracketed interval is a clip-bootstrap 95% interval. Metric-specific `*_n`, `*_n_missing`, finite/infinite counts, and all attempted/successful/failed counts are available in [summary_results.csv]({report_rel}/summary_results.csv).

{_table(summary)}

![Detection rates]({report_rel}/figures/detection_rates.png)

![Message recovery]({report_rel}/figures/message_recovery.png)

![Score distributions]({report_rel}/figures/score_distributions.png)

## Quality, timing, and failure accounting

Across distinct successfully processed clips, preattack watermark SNR was {_numeric_interval(baseline_stats, 'watermark_snr_db')} dB ({len(wm_snr)} clips), watermark STOI was {_numeric_interval(baseline_stats, 'watermark_stoi', 4)} ({len(wm_stoi)} valid clips), and embedding runtime was {_numeric_interval(baseline_stats, 'embed_runtime_ms')} ms/clip. Values are means with clip-bootstrap 95% intervals. The maximum successful-output clipping fraction was {_pct(max_clip)}. Clipping, duration, and runtime are tabulated separately for both branches in the summary CSV.

{runtimes}

The following tables show means [95% clip-bootstrap interval]; NA marks an inapplicable or unavailable metric, and infinite clean-control SNR has no finite interval. Means of attack/runtime measurements include first-use import costs where they occurred. Per-metric denominators and negative-branch STOI, clipping, and duration also appear in the CSV.

{_quality_runtime_tables(summary)}

![Quality and detection]({report_rel}/figures/quality_vs_detection.png)

There were {failures} failed inference/attack rows. [failures.csv]({report_rel}/failures.csv) retains every failed row and its error message, including an empty header-only file if none failed. Setup, download, test, and other command failures are recorded in `STATUS.md` and `outputs/logs/`; absence of failed inference rows does not imply that every setup command succeeded. See those records for exact commands and resolutions.

{setup_failures}

## Interpretation and limits

{attack_text} Differences are descriptive and specific to this speech subset, checkpoint pair, decision threshold, and implemented attack settings. Potential explanations involving codec suppression, resampling bandwidth, pitch/frequency changes, temporal warping, or loss of watermark-bearing segments are hypotheses, not identified mechanisms. Matched controls estimate false positives under the same transformations but cannot characterize rare false positives from {n} controls per condition or a general population. Overlapping conditions reuse the same clips, so {summary.shape[0]} conditions do not create {summary.shape[0]} independent control datasets.

The deterministic selection is not a random sample of all speech. Audiobook speech excludes many languages, recording conditions, music, generated speech, long-form audio, replay channels, adaptive attacks, compound transformations, watermark removal optimization, model shifts, and training-time attacks. This evaluation does not assess adversarial security. Before operational use, a larger speaker-disjoint evaluation, threshold calibration on separate data, broader audio coverage, perceptual listening, and targeted threat-model testing would be needed.

## Reproducibility and audit trail

- Raw input: `{metadata['raw_path']}`; SHA-256 `{metadata['raw_sha256']}`.
- Run ID: `{raw.run_id.iloc[0]}`; recorded raw Git commit(s): `{', '.join(sorted(raw.git_commit.unique())) or 'not recorded'}`.
- Grid: {n} clips × {summary.shape[0]} conditions × 2 matched branches = {len(raw)} rows. All tables and figures in this report were generated from raw per-clip rows, with no manual adjustment of measurements.
- Environment: [`outputs/environment.json`](../outputs/environment.json), resolved dependency lock and provenance records. Environment file was {'present' if environment else 'not present'} when the report was generated; it is the authoritative hardware/software record.
- Dataset composition: [dataset_composition.csv]({report_rel}/dataset_composition.csv); paired contrasts: [paired_differences.csv]({report_rel}/paired_differences.csv); aggregation metadata: [aggregation_metadata.json]({report_rel}/aggregation_metadata.json).
- Recreate aggregates with `.venv/bin/python scripts/generate_report.py --raw {metadata['raw_path']} --output-dir outputs/reproduced_summary --reports-dir reports/reproduced --resamples {metadata['bootstrap_resamples']}`. Output directories must not contain existing generated artifacts.
- Complete run and test commands are in `README.md` and `STATUS.md`. Reproducibility checks compare deterministic messages, attack outputs, and scientific metrics; measured wall-clock runtime is expected to vary.

Validation evidence available when these artifacts were generated is summarized below. Final claims review may follow report creation; its authoritative record is retained separately in `outputs/claims_audit.json` and `STATUS.md`.

{validation_table}
"""
    # Exactly seven rows at most: clean plus the lowest-TPR setting in each
    # transformation family (BER breaks ties). Selection is explicit, descriptive.
    selected = []
    if clean is not None:
        selected.append(clean)
    for _, group in attacks.groupby("attack_family", sort=False):
        selected.append(group.sort_values(["tpr_mean", "ber_mean", "condition"], ascending=[True, False, True], na_position="last").iloc[0])
    panel_table = _table(pd.DataFrame(selected), intervals=False)
    panel_false_positive_note = ""
    if len(zero_conditions):
        panel_false_positive_note = (f"Zero observed FPR gives a degenerate [0,0] bootstrap interval, not proof of zero population FPR; "
                                     f"the largest supplemental 95% Wilson upper bound was {_pct(zero_conditions.fpr_wilson_ci_high.max())}, assuming independent clips.")
    panel = f"""# AudioSeal: preliminary robustness panel

**Scope.** Official pretrained AudioSeal; {n} public LibriSpeech clips, {speakers} speakers, mono 16 kHz, {_number(clips.duration_seconds.min(), 1)}–{_number(clips.duration_seconds.max(), 1)} seconds. Deterministic 16-bit messages; seed {SEED}. {summary.shape[0]} conditions × matched positive/negative branches = {len(raw)} attempted rows. No training or fine-tuning.

**Decision.** Frame watermark probability > {raw.frame_threshold.iloc[0]:g}; clip positive when positive-frame fraction > {raw.detector_threshold.iloc[0]:g}. The clip threshold is study-prespecified, without deployment calibration.

**Observed results.** {clean_text} {attack_text}

Clean and lowest-TPR setting per family are shown; higher BER breaks ties. All settings and 95% intervals appear in the technical report. These are descriptive selections.

{panel_table}

**Quality/runtime.** Mean watermark SNR {_number(wm_snr.mean())} dB; STOI {_number(wm_stoi.mean(), 3)}. Maximum clipping {_pct(max_clip)}. Median embedding {_number(once.embed_runtime_ms.median(), 1)} ms/clip; detection {_number(pd.concat([positive_ok, negative_ok]).detect_runtime_ms.median(), 1)} ms/inference, excluding model loading/warmup. Temporal incompatibility makes sample-aligned SNR/STOI inapplicable. These metrics do not establish inaudibility; no listening study was performed.

**Uncertainty/failures.** {metadata['bootstrap_resamples']:,} clip-bootstrap resamples; bits are not independent samples and conditions reuse paired clips. {failures}/{len(raw)} rows failed; denominators and failure bounds remain in CSV. {panel_false_positive_note}

**Limits.** One checkpoint pair and a small audiobook subset provide a preliminary baseline. Speaker dependence can make clip intervals optimistic. Findings do not establish general robustness, security, or deployment readiness. Next: larger speaker-disjoint tests, independent threshold calibration, broader audio/attacks, and listening tests.

**Evidence.** [Technical report](technical_report.md), [raw-derived summary]({report_rel}/summary_results.csv), [failures]({report_rel}/failures.csv), and [source notes](source_notes.md).
"""
    (reports / "technical_report.md").write_text(technical, encoding="utf-8")
    (reports / "panel_summary.md").write_text(panel, encoding="utf-8")


def generate_report(raw_path: str | Path, output_dir: str | Path, reports_dir: str | Path,
                    resamples: int = 2000, seed: int = SEED) -> dict:
    """Generate immutable report artifacts; all numeric results originate in CSV."""
    if seed != SEED:
        raise ValueError("This protocol fixes seed 20260905.")
    raw_path, output, reports = Path(raw_path), Path(output_dir), Path(reports_dir)
    targets = [output / name for name in ("summary_results.csv", "dataset_composition.csv", "paired_differences.csv", "failures.csv", "aggregation_metadata.json")]
    targets += [output / "figures" / name for name in ("detection_rates.png", "message_recovery.png", "score_distributions.png", "quality_vs_detection.png")]
    targets += [reports / name for name in ("technical_report.md", "panel_summary.md")]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing artifacts: {existing}")
    raw = load_raw(raw_path)
    summary = aggregate_results(raw, resamples, seed)
    paired = paired_changes(raw, resamples, seed)
    composition = dataset_composition(raw)
    metadata = {
        "raw_path": str(raw_path), "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "n_raw_rows": len(raw), "n_clips": int(raw.clip_id.nunique()),
        "n_speakers": int(raw.speaker_id.nunique()), "n_conditions": int(raw.condition.nunique()),
        "n_failed_rows": int((raw.status == "failed").sum()), "bootstrap_seed": seed,
        "bootstrap_resamples": resamples, "bootstrap_unit": "clip",
        "bootstrap_method": "95% percentile interval; paired clip differences for attack-minus-clean",
        "rate_denominator": "successfully measured clips; failed and missing counts explicit",
        "zero_rate_caveat": "A [0,0] empirical bootstrap interval does not establish zero population FPR.",
        "quality_missing": "Inapplicable/unavailable metrics are NA, never zero.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    for frame, name in ((summary, "summary_results.csv"), (paired, "paired_differences.csv"),
                        (composition, "dataset_composition.csv"), (raw.loc[raw.status == "failed"], "failures.csv")):
        frame.to_csv(output / name, index=False, na_rep="NA", float_format="%.12g")
    (output / "aggregation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    _figures(raw, summary, output / "figures")
    _write_reports(raw, summary, paired, metadata, reports, output)
    return metadata
