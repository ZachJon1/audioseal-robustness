# Project status

## A — environment inspection complete

- Workspace originally contains only the user's execution-plan Markdown and empty protected metadata directories. User files preserved; project is in a new subdirectory.
- Python 3.9.12 (Anaconda); system Python 3.8 also available. No Python 3.10/3.11 initially. FFmpeg/ffprobe absent. Git 2.25.1. Ubuntu 20.04.6.
- AMD Ryzen Threadripper PRO 5975WX, 64 logical CPUs, ~220 GiB RAM, ~208 GiB available; ~252 GB free project disk. NVIDIA-SMI fails to communicate with driver; use CPU.
- Commands: `pwd`, `rg --files`, `ls -la`, Python versions, `command -v`, `ffmpeg -version`, `git --version`, `lscpu`, `nvidia-smi`, `df -h . /tmp`, `free -h`, `cat /etc/os-release`.
- Failed commands: `rg --files` returned 1 (no matching scaffolding files); `ffmpeg` missing; `nvidia-smi` driver unavailable; Python glob paths missing; `git status` found no initialized repository; sandbox curl DNS failed for PyPI/GitHub. Network-enabled curl successfully downloaded the uv installer to `/tmp/audioseal-uv-install.sh`.
- Plan: see IMPLEMENTATION_PLAN.md. Next exact action: install uv into `.tools`, download managed Python 3.11 into `.python`, create `.venv`, and install CPU-only pinned research dependencies.

## B — in progress

- Created directories, ignore rules and implementation plan. No global packages changed.
- Expected downloads: local tooling/Python, CPU research dependencies, official checkpoints and 346.7 MB LibriSpeech archive; no single download over 1 GB. Dataset remains smoke-gated.

## B — environment created; C — first smoke stopped and diagnosed

- Local uv 0.12.10, CPython 3.11.16, official AudioSeal 0.2.0, PyTorch/torchaudio 2.5.1+cpu; 58 resolved packages initially; `uv pip check` passed. Dependencies pinned in requirements and transitive versions in requirements-lock.txt. Dependency download log totals about 363 MiB compressed (1.4 GB unpacked cache is not network transfer).
- Commands: local uv installer; `uv python install 3.11`; `uv venv`; `uv pip install ... -r requirements.txt`; `uv pip freeze`; `uv pip check`; `python scripts/fetch_models.py`; `python scripts/run_smoke_test.py`.
- Official immutable revision checkpoints downloaded: generator 58,805,980 bytes, detector 34,667,641 bytes, official example 334,496 bytes. URLs and SHA256 in outputs/model_provenance.json.
- Smoke attempt 1 failed with `torch._dynamo.exc.BackendCompilerFailed: No module named setuptools`, preserved in outputs/logs/smoke_test.log and outputs/smoke_test_failed_01.json. No dataset downloaded.
- Resolution: pin/install setuptools and use official eager inference with `TORCHDYNAMO_DISABLE=1`; no AudioSeal package/model code modifications. Next exact action: repeat smoke into a fresh log, then capture environment.json; dataset gate remains closed until pass.

## B and C — passed

- Commands: `python scripts/run_smoke_test.py > outputs/logs/smoke_test_retry02.log`; `uv pip freeze`; `python scripts/check_environment.py`.
- Smoke evidence: outputs/smoke_test.json status passed. Official 7.584 s example, shape [1,1,121344], positive score 1.0, negative score 0.0000906514, recovered all 16 expected bits, watermark SNR 28.6399 dB, peak 0.6740. High-level official API score/message cross-check passed.
- outputs/environment.json records Python 3.11.16, all 59 installed package versions, CPU mode, disabled optional compilation, deterministic algorithms, local FFmpeg 7.0.2, hardware and checkpoint provenance. Earlier failure retained.
- Next exact action (D): `.venv/bin/python scripts/prepare_data.py --download` (346,663,984-byte official archive, smoke-gated).

## E — implementation verification complete (independent of dataset preparation)

- Implemented real FFmpeg MP3 encode/decode, exact target-SNR Gaussian noise, polyphase resampling, librosa phase-vocoder pitch/stretch, seeded contiguous cropping. Limiting and alignment assumptions recorded.
- Command: `.venv/bin/python -m pytest tests/test_attacks.py tests/test_metrics.py -q` — 59 passed in 16.83 s. Tests include deterministic repeated outputs for all 13 conditions, branch-matched randomness, range/length checks, 20/30 dB SNR, strict message metrics and whole-clip bootstrap.
- Next action: integrate raw schema and experiment runner, then verify completeness before pilot.

## D — passed

- Command: `.venv/bin/python scripts/prepare_data.py --download`; passed in 17.18 s. Archive 346,663,984 bytes; official MD5 matched. No extraction/decode failures. Selected 24 complete utterances from 12 speakers, 2.09–9.815 s, 107.49 s total, from 1,971 eligible utterances. Manifest SHA256 `85a65f1f57cd5918e3529a1210a45ad78cb20cab8a6db637589aecc38e333336`.
- Evidence: data/manifests/evaluation_manifest.csv and outputs/data_preparation.json. Source IDs, expected messages and source/processed hashes preserved. SciPy deterministic WAV serialization avoids libsndfile FLOAT WAV PEAK timestamps found in a reproducibility probe.
- Offline data-helper tests: 15 passed. Source notes disclose inaccessible IEEE full text; official DeepMark repository verified the citation and documented design.

## F — integration and verification

- Implemented paired runner, streaming raw CSV writes, failure rows, official detector adapter, 16-bit comparison, aligned SNR/STOI, durations/clipping/runtime, and 2,000 clip-bootstrap summaries with paired attack-minus-clean differences.
- Command: `.venv/bin/python -m pytest -q > outputs/logs/tests_before_pilot.log` — 84 passed, 14 upstream Matplotlib/pyparsing deprecation warnings. Added further audio I/O, result completeness and separate-process deterministic message tests for the integrated runner; running next.
- Additional failed inspection commands preserved here: pre-initialization `git status` (no Git repository); attempted reads of not-yet-created test files/incorrect adapter.py filename; one inline reporting probe omitted PYTHONPATH and failed `ModuleNotFoundError: audio_wm_eval`, corrected with PYTHONPATH=src. Two subagents reached service usage limits before their second tasks completed; root continued the work. No empirical result or failed clip was removed.
- Next exact actions: run complete pre-pilot tests, initialize/commit only the new project with ignored binary artifacts excluded, then `.venv/bin/python scripts/run_experiment.py --limit 2 --output-dir outputs/pilot --save-audio all`.

## F — passed; G — pilot passed and inspected

- Integrated suite: `.venv/bin/python -m pytest -q > outputs/logs/tests_integrated_before_pilot.log` — 108 passed in 5.64 s, 14 upstream deprecation warnings.
- Initialized Git only inside the new project and committed implementation/manifest (`3a1ad8dfa458b46fc17f0ade0b5f23d3607a71f1`). Verified tracked files contain no audio, checkpoints, caches or environment binaries.
- Pilot command: `.venv/bin/python scripts/run_experiment.py --limit 2 --output-dir outputs/pilot --save-audio all`; 52/52 valid rows, zero failed rows, 30.14 s total. Raw SHA256 `9e6572f39d5df80421682ad86611d01faa8bd13e347126e9b0fd888f9cfd1328`.
- Inspector command: `.venv/bin/python scripts/inspect_pilot.py` — passed; all 52 WAVs decode and have valid non-silent ranges, all target-noise and duration checks pass. Both clean positives detected/recovered all 16 bits, both clean controls negative. Numeric crop/stretch/noise settings match metadata.
- Viewed outputs/pilot/inspection.png across clean + all six families. Waveforms and spectra show expected bandwidth, duration and noise changes. This is objective visual/sample inspection; no human listening was performed or claimed.
- Pilot observations retained: pitch errors; detection/message dissociation under stretching and cropping. These are experimental outcomes, not pipeline failures, and do not invalidate the complete run.
- Estimated full runtime 361.6 s; first-clip full-run audio examples ~7.0 MB. Next exact action: `.venv/bin/python scripts/run_experiment.py --output-dir outputs` after source snapshot; then independent two-clip replay and raw-derived reporting.
