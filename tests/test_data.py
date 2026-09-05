"""Offline tests of the actual dataset preparation and selection helpers."""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
import pytest
import soundfile as sf


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_data_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def flac_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    output = io.BytesIO()
    sf.write(output, samples, sample_rate, format="FLAC")
    return output.getvalue()


def add_member(archive: tarfile.TarFile, name: str, content: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    archive.addfile(member, io.BytesIO(content))


@pytest.fixture
def speech_archive(tmp_path: Path) -> Path:
    """Thirteen synthetic speakers; no public data or pretrained model needed."""
    path = tmp_path / "speech.tar.gz"
    samples = (0.1 * np.sin(np.arange(32000, dtype=np.float32) * 0.04)).astype(np.float32)
    speech = flac_bytes(samples)
    with tarfile.open(path, "w:gz") as archive:
        # Deliberately unsorted archive order must not affect seeded selection.
        for speaker in reversed(range(20, 33)):
            for clip in (2, 0, 1):
                add_member(archive, f"LibriSpeech/test-clean/{speaker}/1/{speaker}-1-{clip:04d}.flac", speech)
        add_member(archive, "LibriSpeech/test-clean/20/1/20-1-0100.flac", flac_bytes(samples[:1000]))
        add_member(archive, "LibriSpeech/test-clean/20/1/20-1-0101.flac", flac_bytes(np.column_stack([samples, samples])))
        add_member(archive, "LibriSpeech/test-clean/20/1/20-1-0102.flac", flac_bytes(samples, 8000))
        add_member(archive, "LibriSpeech/test-clean/README.txt", b"synthetic fixture")
    return path


def test_expected_message_has_stable_golden_value_and_preserves_identity() -> None:
    assert data.message_for_clip("123-45-0000") == "1001111100000000"
    assert data.message_for_clip("123-45-0000") == data.message_for_clip("123-45-0000", 20260905)
    assert data.message_for_clip("123-45-0001") != data.message_for_clip("123-45-0000")
    for clip_id in ("123-45-0000", "123-45-0001", "0"):
        message = data.message_for_clip(clip_id)
        assert isinstance(message, str) and len(message) == 16 and set(message) <= {"0", "1"}


def test_existing_user_file_is_reused_only_if_identical(tmp_path: Path) -> None:
    path = tmp_path / "nested/original.bin"
    data.checked_write(path, b"original contents")
    original_stat = path.stat()
    data.checked_write(path, b"original contents")
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns
    with pytest.raises(FileExistsError, match="refusing overwrite"):
        data.checked_write(path, b"changed contents")
    assert path.read_bytes() == b"original contents"


@pytest.mark.parametrize("evidence", [{}, {"status": "failed"}, {"passed": False}, {"status": "running"}])
def test_smoke_gate_rejects_unpassed_evidence(tmp_path: Path, evidence: dict) -> None:
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(evidence))
    with pytest.raises(RuntimeError, match="has not passed"):
        data.require_smoke_pass(path)


@pytest.mark.parametrize("evidence", [{"status": "passed"}, {"passed": True}])
def test_smoke_gate_accepts_explicit_pass(tmp_path: Path, evidence: dict) -> None:
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(evidence))
    assert data.require_smoke_pass(path) == evidence


def test_main_never_downloads_without_smoke_and_records_failure(tmp_path: Path, monkeypatch) -> None:
    def forbidden_download(*args, **kwargs):
        raise AssertionError("Download was attempted before smoke passed")

    monkeypatch.setattr(data, "PROJECT", tmp_path)
    monkeypatch.setattr(data, "download_archive", forbidden_download)
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--download"])
    assert data.main() == 1
    report = json.loads((tmp_path / "outputs/data_preparation.json").read_text())
    assert report["status"] == "failed"
    assert "passing official smoke evidence" in report["failures"][0]["error"]
    assert not (tmp_path / "data/raw/test-clean.tar.gz").exists()


def test_archive_selection_and_saved_manifest_fields_are_reproducible(speech_archive: Path, tmp_path: Path) -> None:
    report = {"failures": []}
    candidates = data.inspect_archive(speech_archive, report, 2.0, 10.0)
    assert report["failures"] == []
    assert report["selection_counts"]["eligible_clips"] == 39
    assert report["selection_counts"]["excluded_duration"] == 1
    assert report["selection_counts"]["excluded_format"] == 2
    selected = data.choose_clips(candidates, 20260905, 12, 2)
    assert len(selected) == 24
    assert len({clip["clip_id"] for clip in selected}) == 24
    assert set(Counter(clip["speaker_id"] for clip in selected).values()) == {2}
    assert selected[0]["speaker_id"] != selected[1]["speaker_id"]

    # Reordering archive-derived containers and changing content hashes cannot
    # change selection. Selection sees identities/eligibility, never inference.
    reordered = {speaker: [dict(clip, source_sha256="different content, same eligibility")
                           for clip in reversed(clips)]
                 for speaker, clips in reversed(list(candidates.items()))}
    repeated_selection = data.choose_clips(reordered, 20260905, 12, 2)
    assert [clip["clip_id"] for clip in selected] == [clip["clip_id"] for clip in repeated_selection]

    output_root = tmp_path / "project"
    rows = data.save_selected(speech_archive, selected, 20260905, "fixture://speech", output_root, report)
    first_hashes = [clip["processed_sha256"] for clip in rows]
    # Regression check for libsndfile's FLOAT WAV PEAK timestamp: a later run
    # must yield the same bytes, not merely the same decoded samples.
    time.sleep(1.05)
    repeated = data.save_selected(speech_archive, selected, 20260905, "fixture://speech", output_root, report)
    assert rows == repeated and first_hashes == [clip["processed_sha256"] for clip in repeated]
    assert report["failures"] == []
    for clip in rows:
        assert set(clip) == set(data.FIELDS)
        assert clip["seed"] == 20260905 and clip["source_split"] == "test-clean"
        output = output_root / clip["processed_path"]
        decoded, sample_rate = sf.read(output, dtype="float32", always_2d=True)
        assert sample_rate == 16000 and decoded.shape == (32000, 1)
        assert np.isfinite(decoded).all() and np.max(np.abs(decoded)) <= 1.0
        assert clip["duration_seconds"] == 2.0
        assert hashlib.sha256(output.read_bytes()).hexdigest() == clip["processed_sha256"]
        source = output_root / "data/raw/selected" / clip["source_filename"]
        original, _ = sf.read(source, dtype="float32", always_2d=True)
        assert np.array_equal(original, decoded)
        assert data.digest_file(source) == clip["source_sha256"]
        assert clip["expected_message"] == data.message_for_clip(clip["clip_id"])


def test_selected_decode_failure_is_recorded_without_replacement(speech_archive: Path, tmp_path: Path) -> None:
    report = {"failures": []}
    candidates = data.inspect_archive(speech_archive, report, 2.0, 10.0)
    selected = data.choose_clips(candidates, 20260905, 12, 2)
    selected[0] = dict(selected[0], source_sha256="mismatch")
    with pytest.raises(RuntimeError, match="23/24 complete; no replacement"):
        data.save_selected(speech_archive, selected, 20260905, "fixture://speech", tmp_path / "project", report)
    assert len(report["failures"]) == 1
    assert report["failures"][0]["member"] == selected[0]["source_filename"]
    assert "changed since metadata inspection" in report["failures"][0]["error"]
    assert len(list((tmp_path / "project/data/processed").glob("*.wav"))) == 23


def test_unsafe_paths_and_link_members_are_rejected() -> None:
    for name in ("../../outside.flac", "/LibriSpeech/test-clean/20/1/20-1-0000.flac",
                 "LibriSpeech/test-clean/20/../20-1-0000.flac"):
        member = tarfile.TarInfo(name)
        member.size = 100
        assert not data.safe_flac_member(member)
    member = tarfile.TarInfo("LibriSpeech/test-clean/20/1/20-1-0000.flac")
    member.size = 100
    member.type = tarfile.SYMTYPE
    assert not data.safe_flac_member(member)
    member.type = tarfile.LNKTYPE
    assert not data.safe_flac_member(member)


def test_invalid_archive_audio_is_visible_in_failure_record(tmp_path: Path) -> None:
    path = tmp_path / "bad.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        add_member(archive, "../../outside.flac", b"untrusted member")
        add_member(archive, "LibriSpeech/test-clean/20/1/20-1-0000.flac", b"invalid FLAC")
    report = {"failures": []}
    assert not data.inspect_archive(path, report, 2.0, 10.0)
    assert report["selection_counts"]["inspection_failures"] == 2
    assert len(report["failures"]) == 2
    assert {row["stage"] for row in report["failures"]} == {"inspect_archive"}
    assert not (tmp_path / "outside.flac").exists()


def test_download_budget_rejects_large_declared_body_without_reading(tmp_path: Path, monkeypatch) -> None:
    class Response(io.BytesIO):
        headers = {"Content-Length": str(data.MAX_DOWNLOAD_BYTES + 1)}

        def read(self, *args):
            raise AssertionError("An over-budget body must never be read")

    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    path = tmp_path / "archive.tar.gz"
    report = {}
    with pytest.raises(RuntimeError, match="exceeds 400 MB budget"):
        data.download_archive(path, data.SOURCE_URL, report)
    assert not path.exists()
    assert report["download"]["bytes_received"] == 0


def test_incomplete_download_is_preserved_and_not_hidden(tmp_path: Path, monkeypatch) -> None:
    class Response(io.BytesIO):
        headers = {"Content-Length": "10"}

    monkeypatch.setattr(data.urllib.request, "urlopen", lambda *args, **kwargs: Response(b"short"))
    path = tmp_path / "archive.tar.gz"
    report = {}
    with pytest.raises(RuntimeError, match="Incomplete download"):
        data.download_archive(path, data.SOURCE_URL, report)
    assert path.read_bytes() == b"short"
    assert report["download"]["bytes_received"] == 5
