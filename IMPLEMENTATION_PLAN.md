# Implementation plan

Use official PyPI AudioSeal and the official non-streaming 16-bit generator/detector checkpoints, exclusively for pretrained CPU inference. Create a local Python 3.11 virtual environment, use CPU PyTorch wheels, and obtain FFmpeg through imageio-ffmpeg. Record resolved packages, source/checkpoint hashes, hardware, command failures, and run provenance.

Execute gates in order: A inspect; B scaffold/environment; C official test.wav smoke (positive and negative); D deterministic 24-clip LibriSpeech test-clean manifest (12 speakers, two utterances each, 2–10 s); E deterministic attack tests; F metrics/statistics/schema tests; G two-clip complete pilot and numerical/audio inspection; H complete 624-row run; I raw-derived CSVs/figures; J technical report and panel summary; K complete tests and fresh-process reproducibility/claims audit.

The 13 conditions are clean and two settings each for MP3, Gaussian noise, sample-rate round trips, pitch shift, stretch and crop. Every condition has matched watermarked/unwatermarked branches. Messages and attack seeds derive deterministically from seed 20260905 and stable clip identifiers. No selection or threshold tuning uses detector performance.

Detection and recovery are separate endpoints. Record AudioSeal's fraction of frames above its default 0.5 posterior threshold; predeclare a clip decision rule following the official example. Bootstrap whole clips with 2,000 resamples and preserve branch/condition pairing. Describe speaker dependence and boundary-degenerate bootstrap intervals. Record failed combinations and missing/inapplicable metrics explicitly. Evaluate pre-attack watermark SNR only on aligned signals; exclude temporal alterations from sample-aligned quality calculations. STOI is optional and limited to valid aligned pairs; no listening or perceptual-quality claims without human evidence.

Stop at invalid gates, unavailable official weights, unresolved metric definitions, downloads above 1 GB, or required privileged system installation. Preserve all pre-existing files and refuse result overwrite. Generated reports derive numerical claims from raw rows.
