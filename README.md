# audioseal-robustness

Preliminary robustness evaluation of **official pretrained Meta AudioSeal 0.2.0** on 24 public LibriSpeech speech clips from 12 speakers. No training or fine-tuning. This small, single-system experiment cannot establish general audio-watermarking robustness.

## Environment and setup

Python 3.11.16, CPU PyTorch/torchaudio 2.5.1, and FFmpeg 7.0.2 bundled through imageio-ffmpeg. All resolved package versions are in `requirements-lock.txt`; hardware, software, FFmpeg hash and model provenance are in `outputs/environment.json`. Nothing installs globally. A usable GPU driver was unavailable on the execution host. Official eager inference is used with optional PyTorch compilation disabled.

For a new checkout, create a Python 3.11 virtual environment and install the locked dependencies:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-lock.txt
```

If Python 3.11 is absent, `scripts/setup.sh` installs pinned uv and managed Python inside this project. Initial compressed transfers including dataset and checkpoints are approximately 0.9 GB total; unpacked local storage is larger. No system installation is needed. Inspect the script before running it. Downloads, managed Python, virtual environment, caches, checkpoints and generated audio are ignored by Git.

This repository includes the recorded results. For an entirely new execution **in a separate checkout**, first preserve that reference evidence; the commands below require previously unused output paths:

```bash
mkdir reference-run
mv outputs reports reference-run/
mkdir outputs reports
cp reference-run/reports/source_notes.md reports/
```

Then execute the original smoke-to-pilot sequence. The smoke gate will run anew before the dataset download. On this existing working copy, the alternative commands below reuse verified inputs without moving evidence.

```bash
bash scripts/setup.sh
.venv/bin/python scripts/fetch_models.py
.venv/bin/python scripts/run_smoke_test.py
.venv/bin/python scripts/check_environment.py
.venv/bin/python scripts/prepare_data.py --download
.venv/bin/python -m pytest -q
git add .
git commit -m "Record evaluation implementation and manifest"
.venv/bin/python scripts/run_experiment.py --limit 2 --output-dir outputs/pilot --save-audio all
.venv/bin/python scripts/inspect_pilot.py
```

Inspect `outputs/pilot/raw_results.csv`, `validation.json`, `inspection.json`, `inspection.png`, and the saved WAV examples before a full run. The numerical gate checks every branch/condition, clean performance, target SNR, duration changes, audio ranges and row completeness. Human listening is an additional useful assessment; this project does not claim to have performed it.

```bash
.venv/bin/python scripts/run_experiment.py --output-dir outputs
.venv/bin/python scripts/generate_report.py --raw outputs/raw_results.csv --output-dir outputs --reports-dir reports
.venv/bin/python scripts/run_experiment.py --limit 2 --output-dir outputs/reproducibility --save-audio none
.venv/bin/python scripts/compare_runs.py --reference outputs/pilot/raw_results.csv --candidate outputs/reproducibility/raw_results.csv --output outputs/reproducibility/comparison.json
.venv/bin/python scripts/run_tests.py
.venv/bin/python scripts/audit_project.py
```

The completed outputs are [technical report](reports/technical_report.md), [panel summary](reports/panel_summary.md), [raw results](outputs/raw_results.csv), [summary results](outputs/summary_results.csv), and [claims audit](outputs/claims_audit.json). The final audit recomputes every summary cell, validates the entire grid and provenance, and checks replay evidence. Its command is `.venv/bin/python scripts/audit_project.py`; to repeat it without replacing evidence, use `--output outputs/another_claims_audit.json`.

Existing results are never silently overwritten. In this completed workspace, use fresh output directories to reproduce inference and summaries from the verified local inputs:

```bash
.venv/bin/python scripts/run_experiment.py --output-dir outputs/another_run --save-audio none
.venv/bin/python scripts/generate_report.py --raw outputs/raw_results.csv --output-dir outputs/another_summary --reports-dir reports/another_summary
.venv/bin/python scripts/run_tests.py --output-dir outputs/another_test_run
```

Smoke/environment/inspection scripts refuse existing evidence; use the separate-checkout sequence above for a new end-to-end gated run. Model fetch verifies existing hashes; data preparation verifies existing files and regenerates the identical manifest without overwriting it. Scripts insert `src` in the import path; interactive Python requires `PYTHONPATH=src` or an editable project installation.

## Design and metric definitions

Seed **20260905**. `configs/attacks.yaml` defines 13 conditions: clean and two settings each of MP3, Gaussian noise, resampling, pitch, stretch and cropping. Apply each to both original and watermarked audio: **624 raw rows**. Random noise realizations and crop locations use the same derived seed across branches; noise amplitude is scaled to each branch's RMS. Crop retains one seeded contiguous segment of 90%/75% duration. Stretch rate r gives duration approximately T/r. All detector inputs are mono16 kHz float32. Final attack limiting is recorded; embedding that exceeds the valid range is a failure.

The manifest records a deterministic 16-bit expected payload per clip, independent of detector output. Selection uses full 2–10 s utterances and fixed random sampling of 12 speakers then two clips each. Data CLI defaults match `configs/experiment.yaml`; use its explicit selection options when changing the data design. Selection never uses model results.

- **Detection score:** official fraction of frames with positive posterior >0.5. **Detected:** study-prespecified score >0.5. The clip threshold is not represented as a calibrated official default. Mean frame probability is additionally retained. No threshold tuning.
- **Recovery:** compare all 16 bits on every positive row, including undetected positives. BER, bit accuracy and exact recovery are distinct from presence detection. Negative-control recovery metrics are NA because no message was embedded.
- **Quality:** watermark SNR/STOI compare original against clean marked audio, once per clip. Post-attack quality compares aligned outputs to original. MP3 alignment is unverified; pitch/stretch/crop are excluded from sample-aligned SNR/STOI. `NA` and `quality_applicability` mark missing/inapplicable values; identical signals can have infinite SNR. PESQ/ViSQOL omitted; valid aligned STOI is included. Metrics do not establish inaudibility or human intelligibility.
- **Uncertainty:** 2,000 whole-clip bootstrap resamples, never independent bits. Repeated speakers and small n limit inference; zero observed errors can have degenerate bootstrap intervals. Supplemental Wilson bounds are descriptive, also subject to dependence.
- **Runtime:** CPU wall time; loading and warmup separate. Embed time measured once per clip and repeated in rows, so never sum it over conditions. Attacks include codec I/O and initial imports. Exact audio sample hashes and scientific outputs are reproducibility targets; wall-clock times vary.

Raw rows are flushed before aggregation. Failed combinations remain visible with tracebacks. Every summary and figure is generated from raw rows; reports disclose source-access failures and operational failures. `STATUS.md` contains phase evidence and unresolved limitations. Run manifests record the Git commit, actual code/config/manifest hashes and dirty status.

## Sources

[AudioSeal paper](https://proceedings.mlr.press/v235/san-roman24a.html), [official implementation](https://github.com/facebookresearch/audioseal), [SoK robustness framing](https://arxiv.org/abs/2503.19176), [DeepMark design reference](https://doi.org/10.1109/ACCESS.2026.3685903), [LibriSpeech/OpenSLR12](https://www.openslr.org/12/). Source-specific access limitations and dataset attribution are in `reports/source_notes.md` and `data/README.md`.
