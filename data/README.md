# Evaluation data

The initial evaluation uses 24 complete utterances from 12 speakers in the public
[LibriSpeech test-clean split](https://www.openslr.org/12). LibriSpeech is read
English audiobook speech, distributed by OpenSLR under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute the corpus to
Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur,
*LibriSpeech: an ASR corpus based on public domain audio books*, ICASSP 2015,
pp. 5206–5210. This small subset does not represent other languages, music, noisy
speech, all speakers, or production audio.

Run from the project directory, only after the official one-file smoke test passes:

```bash
.venv/bin/python scripts/prepare_data.py --download
```

The official archive is approximately 346.7 MB. The downloader caps transfers at
400 MB and verifies MD5 `32fa31d27d2e1cad72775fee3f4849a9` from
[OpenSLR's published checksums](https://www.openslr.org/resources/12/md5sum.txt).
It also records SHA-256. Use `--archive /path/to/test-clean.tar.gz` for an existing
archive, or `--url` with an official OpenSLR mirror. The archive must pass the same
checksum whichever mirror is used. The script refuses dataset preparation until
`outputs/smoke_test.json` reports `status: passed` or `passed: true`.

Selection is independent of AudioSeal outcomes. All regular, safely named FLAC
members are inspected for mono 16 kHz audio with a duration of 2–10 seconds.
Speakers with at least two eligible utterances are sorted. NumPy's default RNG,
seeded with `20260905`, samples 12 speakers without replacement, then samples two
clips from each speaker's sorted eligible clip list without replacement. Manifest
order interleaves speakers so the two-clip pilot includes two speakers. No source
utterance is cropped, normalized, amplified, resampled, or replaced based on a
watermark result. Metadata or decode failures stop preparation and are recorded.

The manifest preserves the source archive member name, source URL, source FLAC
SHA-256, processed WAV SHA-256, speaker, sample rate, duration, sample count, seed,
and deterministic expected 16-bit message. Message seed material is UTF-8
`20260905:{clip_id}:message`; the first eight SHA-256 digest bytes, interpreted as
a big-endian integer, seed `numpy.random.default_rng`, which samples 16 binary
integers. Read `expected_message` as a string to preserve leading zeros.

`processed_path` is relative to the project directory. Processed files are
32-bit FLOAT WAV and are verified sample-identical to decoded FLAC float32 audio.
SciPy serializes these files without the changing PEAK timestamp emitted by the
libsndfile float-WAV writer, allowing file hashes to remain stable across reruns.
Original selected FLAC members are retained beneath `raw/selected/`. The archive,
source audio, processed audio, and generated experiment audio stay outside Git.
The small CSV manifest stays in Git so the selected identities and hashes can be
reviewed. Existing files are reused only if their content matches exactly; no
pre-existing file is overwritten or deleted by this script.

`outputs/data_preparation.json` records selection counts, durations, archive hash,
command, runtime, and failures. Reruns produce an additional uniquely named JSON
audit record. Repeating the command against the same verified archive must yield
the identical CSV manifest and audio hashes.
