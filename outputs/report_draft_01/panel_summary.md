# AudioSeal: preliminary robustness panel

**Scope.** Official pretrained AudioSeal; 24 public LibriSpeech clips, 12 speakers, mono 16 kHz, 2.1–9.8 seconds. Deterministic 16-bit messages; seed 20260905. 13 conditions × matched positive/negative branches = 624 attempted rows. No training or fine-tuning.

**Decision.** Frame watermark probability > 0.5; clip positive when positive-frame fraction > 0.5. The clip threshold is study-prespecified, without deployment calibration.

**Observed results.** Clean TPR was 100.0% [100.0%, 100.0%]; FPR was 0.0% [0.0%, 0.0%]. Clean BER was 0.0% [0.0%, 0.0%] and exact 16-bit recovery was 100.0% [100.0%, 100.0%]. Among tested transformations, the lowest measured TPR was 0.0% for Pitch −2 st (BER 51.3%, exact recovery 0.0%). Ties are visible in the full table.

Clean and lowest-TPR setting per family are shown; higher BER breaks ties. All settings and 95% intervals appear in the technical report. These are descriptive selections.

| Condition | TPR | FPR | BER | Exact 16-bit | Failed rows |
|---|---:|---:|---:|---:|---:|
| Clean | 100.0% | 0.0% | 0.0% | 100.0% | 0 |
| MP3 128 kbps | 100.0% | 0.0% | 0.0% | 100.0% | 0 |
| Noise 20 dB | 33.3% | 0.0% | 2.3% | 66.7% | 0 |
| Resample 12 kHz | 100.0% | 0.0% | 0.0% | 100.0% | 0 |
| Pitch +2 st | 0.0% | 0.0% | 53.1% | 0.0% | 0 |
| Stretch 0.9× | 8.3% | 0.0% | 2.3% | 75.0% | 0 |
| Crop 25% | 70.8% | 0.0% | 31.2% | 4.2% | 0 |

**Quality/runtime.** Mean watermark SNR 27.60 dB; STOI 0.998. Maximum clipping 0.0%. Median embedding 1427.7 ms/clip; detection 64.5 ms/inference, excluding model loading/warmup. Temporal incompatibility makes sample-aligned SNR/STOI inapplicable. These metrics do not establish inaudibility; no listening study was performed.

**Uncertainty/failures.** 2,000 clip-bootstrap resamples; bits are not independent samples and conditions reuse paired clips. 0/624 rows failed; denominators and failure bounds remain in CSV. Zero observed FPR gives a degenerate [0,0] bootstrap interval, not proof of zero population FPR; the largest supplemental 95% Wilson upper bound was 13.8%, assuming independent clips.

**Limits.** One checkpoint pair and a small audiobook subset provide a preliminary baseline. Speaker dependence can make clip intervals optimistic. Findings do not establish general robustness, security, or deployment readiness. Next: larger speaker-disjoint tests, independent threshold calibration, broader audio/attacks, and listening tests.

**Evidence.** [Technical report](technical_report.md), [raw-derived summary](../outputs/summary_results.csv), [failures](../outputs/failures.csv), and [source notes](source_notes.md).
