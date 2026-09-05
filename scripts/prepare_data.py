#!/usr/bin/env python3
"""Prepare a fixed, model-independent LibriSpeech test-clean evaluation subset.

The only network operation is an explicit --download after a passing smoke test.
Full source utterances are selected using file metadata, never detector outcomes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
import urllib.request
import uuid

import numpy as np
import soundfile as sf
from scipy.io import wavfile


PROJECT = Path(__file__).resolve().parents[1]
SOURCE_URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
EXPECTED_MD5 = "32fa31d27d2e1cad72775fee3f4849a9"
MAX_DOWNLOAD_BYTES = 400_000_000
SEED = 20260905
FIELDS = [
    "clip_id", "speaker_id", "source_split", "source_filename", "source_url",
    "source_sha256", "processed_path", "processed_sha256", "original_sample_rate",
    "sample_rate", "duration_seconds", "num_samples", "seed", "expected_message",
]


def digest_file(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def message_for_clip(clip_id: str, seed: int = SEED) -> str:
    digest = hashlib.sha256(f"{seed}:{clip_id}:message".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return "".join(str(int(bit)) for bit in rng.integers(0, 2, 16))


def checked_write(path: Path, data: bytes) -> None:
    """Allow identical reruns; do not overwrite any pre-existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if digest_file(path) != hashlib.sha256(data).hexdigest():
            raise FileExistsError(f"Existing file differs; refusing overwrite: {path}")
        return
    with path.open("xb") as handle:
        handle.write(data)


def require_smoke_pass(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Dataset preparation requires passing official smoke evidence: {path}")
    evidence = json.loads(path.read_text())
    if evidence.get("status") != "passed" and evidence.get("passed") is not True:
        raise RuntimeError(f"Official smoke test has not passed: {path}")
    return evidence


def download_archive(path: Path, url: str, report: dict) -> None:
    """Bound archive transfer below 1 GB, preserving failed transfers for audit."""
    if path.exists():
        report["download"] = {"status": "existing_archive_used", "url": url}
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"status": "started", "url": url, "bytes_received": 0,
             "max_bytes": MAX_DOWNLOAD_BYTES}
    report["download"] = state
    request = urllib.request.Request(url, headers={"User-Agent": "audioseal-robustness/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        declared_length = response.headers.get("Content-Length")
        state["content_length"] = int(declared_length) if declared_length else None
        if declared_length and int(declared_length) > MAX_DOWNLOAD_BYTES:
            raise RuntimeError("Archive exceeds 400 MB budget; no body downloaded")
        with path.open("xb") as handle:
            while True:
                chunk = response.read(min(1 << 20, MAX_DOWNLOAD_BYTES - state["bytes_received"] + 1))
                if not chunk:
                    break
                state["bytes_received"] += len(chunk)
                if state["bytes_received"] > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError("Download exceeded 400 MB budget; incomplete file preserved")
                handle.write(chunk)
        if declared_length and state["bytes_received"] != int(declared_length):
            raise RuntimeError("Incomplete download; partial archive preserved")
    state["status"] = "completed"


def safe_flac_member(member: tarfile.TarInfo) -> bool:
    parts = PurePosixPath(member.name).parts
    return (
        member.isfile() and not PurePosixPath(member.name).is_absolute()
        and ".." not in parts and len(parts) == 5
        and parts[:2] == ("LibriSpeech", "test-clean")
        and parts[2].isdigit() and parts[3].isdigit()
        and parts[4].endswith(".flac")
        and parts[4].startswith(f"{parts[2]}-{parts[3]}-")
        and 0 < member.size <= 20_000_000
    )


def inspect_archive(archive: Path, report: dict, min_seconds: float, max_seconds: float) -> dict:
    candidates: dict[str, list[dict]] = defaultdict(list)
    counts = Counter()
    seen = set()
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.name.endswith(".flac"):
                counts["non_audio_members"] += 1
                continue
            counts["flac_members"] += 1
            try:
                if not safe_flac_member(member):
                    raise ValueError("Unsafe or unexpected FLAC tar member")
                if member.name in seen:
                    raise ValueError("Duplicate FLAC tar member")
                seen.add(member.name)
                handle = tar.extractfile(member)
                if handle is None:
                    raise ValueError("Unable to read regular FLAC member")
                content = handle.read()
                info = sf.info(io.BytesIO(content))
                duration = info.frames / info.samplerate
                if info.samplerate != 16000 or info.channels != 1:
                    counts["excluded_format"] += 1
                    continue
                if not min_seconds <= duration <= max_seconds:
                    counts["excluded_duration"] += 1
                    continue
                speaker = PurePosixPath(member.name).parts[2]
                candidates[speaker].append({
                    "source_filename": member.name,
                    "clip_id": PurePosixPath(member.name).stem,
                    "speaker_id": speaker,
                    "num_samples": info.frames,
                    "sample_rate": info.samplerate,
                    "duration_seconds": duration,
                    "source_sha256": hashlib.sha256(content).hexdigest(),
                })
                counts["eligible_clips"] += 1
            except Exception as error:
                counts["inspection_failures"] += 1
                report["failures"].append({"stage": "inspect_archive", "member": member.name,
                                           "error": f"{type(error).__name__}: {error}"})
    report["selection_counts"] = dict(counts)
    report["eligible_per_speaker"] = dict(sorted((key, len(value)) for key, value in candidates.items()))
    return candidates


def choose_clips(candidates: dict, seed: int, speakers: int, clips_per_speaker: int) -> list[dict]:
    eligible_speakers = sorted(key for key, value in candidates.items() if len(value) >= clips_per_speaker)
    if len(eligible_speakers) < speakers:
        raise RuntimeError(f"Need {speakers} eligible speakers; found {len(eligible_speakers)}")
    rng = np.random.default_rng(seed)
    selected_speakers = rng.choice(eligible_speakers, size=speakers, replace=False).tolist()
    selected = []
    for speaker in selected_speakers:
        options = sorted(candidates[speaker], key=lambda row: row["clip_id"])
        indices = rng.choice(len(options), size=clips_per_speaker, replace=False)
        selected.extend(options[int(index)] for index in indices)
    # Interleave speaker selections: the first two pilot clips have different speakers.
    return [selected[speaker * clips_per_speaker + clip]
            for clip in range(clips_per_speaker) for speaker in range(speakers)]


def save_selected(archive: Path, selected: list[dict], seed: int, source_url: str,
                  project: Path, report: dict) -> list[dict]:
    wanted = {row["source_filename"]: row for row in selected}
    completed = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if member.name not in wanted:
                continue
            row = dict(wanted[member.name])
            try:
                if not safe_flac_member(member):
                    raise ValueError("Unsafe selected archive member")
                handle = tar.extractfile(member)
                if handle is None:
                    raise ValueError("Selected FLAC cannot be read")
                source = handle.read()
                if hashlib.sha256(source).hexdigest() != row["source_sha256"]:
                    raise ValueError("Source member changed since metadata inspection")
                audio, sample_rate = sf.read(io.BytesIO(source), dtype="float32", always_2d=True)
                if sample_rate != 16000 or audio.shape != (row["num_samples"], 1):
                    raise ValueError(f"Decoded shape/rate mismatch: {audio.shape}, {sample_rate}")
                if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1.0:
                    raise ValueError("Nonfinite or out-of-range source audio")
                if np.mean(audio.astype(np.float64) ** 2) <= 0:
                    raise ValueError("Silent source audio")
                original_path = project / "data" / "raw" / "selected" / member.name
                checked_write(original_path, source)
                processed_path = Path("data") / "processed" / f"{row['clip_id']}.wav"
                buffer = io.BytesIO()
                # libsndfile's FLOAT WAV writer embeds a wall-clock PEAK timestamp.
                # SciPy's IEEE float writer gives stable file hashes across reruns.
                wavfile.write(buffer, sample_rate, audio)
                checked_write(project / processed_path, buffer.getvalue())
                recovered, recovered_rate = sf.read(project / processed_path, dtype="float32", always_2d=True)
                if recovered_rate != sample_rate or not np.array_equal(audio, recovered):
                    raise ValueError("FLOAT WAV is not sample-identical to decoded source")
                row.update({"source_split": "test-clean", "source_url": source_url,
                            "processed_path": processed_path.as_posix(),
                            "processed_sha256": digest_file(project / processed_path),
                            "original_sample_rate": sample_rate, "seed": seed,
                            "expected_message": message_for_clip(row["clip_id"], seed)})
                completed[member.name] = row
            except Exception as error:
                report["failures"].append({"stage": "save_selected", "member": member.name,
                                           "error": f"{type(error).__name__}: {error}"})
    if len(completed) != len(selected):
        raise RuntimeError(f"Selected clip preparation failed: {len(completed)}/{len(selected)} complete; no replacement")
    return [completed[row["source_filename"]] for row in selected]


def write_report(report: dict, outputs: Path) -> Path:
    outputs.mkdir(parents=True, exist_ok=True)
    path = outputs / "data_preparation.json"
    if path.exists():
        path = outputs / f"data_preparation_{uuid.uuid4().hex[:12]}.json"
    with path.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download official 346.7 MB archive after smoke pass")
    parser.add_argument("--archive", type=Path, default=PROJECT / "data/raw/test-clean.tar.gz")
    parser.add_argument("--url", default=SOURCE_URL, help="Official OpenSLR mirror; archive MD5 remains enforced")
    parser.add_argument("--smoke-result", type=Path, default=PROJECT / "outputs/smoke_test.json")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--speakers", type=int, default=12)
    parser.add_argument("--clips-per-speaker", type=int, default=2)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    args = parser.parse_args()
    started = time.perf_counter()
    report = {
        "status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv, "seed": args.seed, "source_url": args.url,
        "source_license": "CC BY 4.0", "source_split": "test-clean",
        "selection": {"speakers": args.speakers, "clips_per_speaker": args.clips_per_speaker,
                      "min_seconds": args.min_seconds, "max_seconds": args.max_seconds,
                      "policy": "full utterances, mono 16 kHz, seeded random speakers then clips; no model outcomes"},
        "failures": [],
    }
    code = 1
    try:
        if args.seed != SEED:
            raise ValueError(f"Initial project requires fixed seed {SEED}")
        if args.speakers <= 0 or args.clips_per_speaker <= 0:
            raise ValueError("Speaker and clip counts must be positive")
        if not 20 <= args.speakers * args.clips_per_speaker <= 30:
            raise ValueError("Initial evaluation requires 20-30 clips")
        if not 2.0 <= args.min_seconds <= args.max_seconds <= 10.0:
            raise ValueError("Initial evaluation requires complete 2-10 second utterances")
        require_smoke_pass(args.smoke_result)
        report["smoke_result_sha256"] = digest_file(args.smoke_result)
        if args.download:
            download_archive(args.archive, args.url, report)
        if not args.archive.is_file():
            raise FileNotFoundError(f"Archive absent: {args.archive}; use --download after smoke test")
        report["archive"] = {"path": str(args.archive), "size_bytes": args.archive.stat().st_size,
                             "md5": digest_file(args.archive, "md5"),
                             "sha256": digest_file(args.archive)}
        if report["archive"]["md5"] != EXPECTED_MD5:
            raise ValueError("Archive MD5 mismatch against official OpenSLR md5sum.txt; file preserved")
        candidates = inspect_archive(args.archive, report, args.min_seconds, args.max_seconds)
        if report["failures"]:
            raise RuntimeError("Archive inspection failures must be reviewed before proceeding")
        selected = choose_clips(candidates, args.seed, args.speakers, args.clips_per_speaker)
        report["selected_clip_ids"] = [row["clip_id"] for row in selected]
        report["selected_speaker_ids"] = sorted({row["speaker_id"] for row in selected})
        rows = save_selected(args.archive, selected, args.seed, args.url, PROJECT, report)
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        manifest = PROJECT / "data/manifests/evaluation_manifest.csv"
        checked_write(manifest, output.getvalue().encode())
        report.update({"status": "passed", "manifest": str(manifest.relative_to(PROJECT)),
                       "manifest_sha256": digest_file(manifest), "num_clips": len(rows),
                       "num_speakers": len({row["speaker_id"] for row in rows}),
                       "total_duration_seconds": sum(row["duration_seconds"] for row in rows),
                       "min_duration_seconds": min(row["duration_seconds"] for row in rows),
                       "max_duration_seconds": max(row["duration_seconds"] for row in rows)})
        code = 0
    except Exception as error:
        report["status"] = "failed"
        report["failures"].append({"stage": "main", "error": f"{type(error).__name__}: {error}"})
        print(report["failures"][-1]["error"], file=sys.stderr)
    finally:
        report["runtime_seconds"] = time.perf_counter() - started
        report_path = write_report(report, PROJECT / "outputs")
        print(f"Data preparation {report['status']}; audit record: {report_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
