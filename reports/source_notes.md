# Source verification notes

Accessed 2026-09-05. These notes distinguish verified primary sources from material
that could not be read. They support a preliminary evaluation of one pretrained
system; they do not establish general robustness or a replication of prior papers.

| Reference | Verified identity and use | Access limitation |
| --- | --- | --- |
| [AudioSeal paper](https://proceedings.mlr.press/v235/san-roman24a.html) | Robin San Roman, Pierre Fernandez, Hady Elsahar, Alexandre Défossez, Teddy Furon, and Tuan Tran. *Proactive Detection of Voice Cloning with Localized Watermarking*. ICML 2024, PMLR 235:43180–43196. Official model and localized watermarking context. | Publisher abstract and bibliographic record read; original paper's numerical benchmark claims are not transferred to this experiment. |
| [Official implementation](https://github.com/facebookresearch/audioseal) | Meta's `facebookresearch/audioseal` repository documents additive watermark generation, optional 16-bit payload, official model cards `audioseal_wm_16bits` and `audioseal_detector_16bits`, and batch × channel × time input. | Current documentation may differ from a pinned installed release; the local adapter must audit installed source and record resolved versions/checkpoint hashes. |
| [Robustness framing](https://arxiv.org/abs/2503.19176v2) | Yizhu Wen, Ashwin Innuganti, Aaron Bien Ramos, Hanqing Guo, and Qiben Yan. *SoK: How Robust is Audio Watermarking in Generative AI models?* arXiv:2503.19176v2, revised 27 March 2025. Supports considering diverse distortions, message recovery, detection reliability, and fidelity as separate outcomes. [Full HTML](https://arxiv.org/html/2503.19176v2) was read selectively. | The current small signal-transformation experiment is much narrower than that study's 109 configurations and does not reproduce its results or all threat models. |
| [Benchmark design reference](https://doi.org/10.1109/ACCESS.2026.3685903) | Slavko Kovačević, Elena Nešović, Kosta Pavlović, Petar Nedić, and Igor Djurović. *DeepMark Benchmark: Redefining Audio Watermarking Robustness*. IEEE Access 14:62031–62044 (2026). Bibliographic identity and DOI are independently provided by the authors' [official benchmark repository](https://github.com/deepmark/deepmarkpy-benchmark). | DOI resolver returned an internal error; [IEEE document 11488564](https://ieeexplore.ieee.org/document/11488564) exposed no usable article text. The 2026 paper's full text was not verified. Design observations below come from the official repository; no claim of reproducing the 2026 paper is made. |
| [LibriSpeech](https://www.openslr.org/12) | OpenSLR SLR12, 16 kHz read English audiobook speech, CC BY 4.0; `test-clean.tar.gz` is listed as approximately 346 MB. [Published checksum](https://www.openslr.org/resources/12/md5sum.txt): `32fa31d27d2e1cad72775fee3f4849a9`. | A fixed small speaker-balanced subset is selected; it cannot characterize corpus-wide or demographic performance. |

The official DeepMark repository documents modular configurable attacks, a clean
embedding-only baseline, separate detection-reliability evaluation with matched
transformed negative controls, and raw per-file outputs. Its documentation also
warns that timing shifts invalidate interpretations of samplewise comparison
metrics as audio quality. This project follows the user's stricter rule: temporal
misalignment or incompatible lengths make sample-aligned SNR explicitly
inapplicable. Parameter values and statistical decisions belong to this project's
prespecified configuration; they are not presented as copied from a paper.

The 2025 ICLR workshop paper with the same DeepMark title has a different author
list, including Murilo Z. Silvestre. It is a distinct earlier version and is not
silently substituted for the requested 2026 IEEE article.

Access failures are retained here: direct Python `urllib.request.urlopen` probes
of OpenSLR's checksum text, IEEE document 11488564, and Crossref's DOI metadata
endpoint failed with `URLError: [Errno -2] Name or service not known` under the
initial network sandbox. Web-tool reads of the official OpenSLR and benchmark
repository pages succeeded. No dataset was downloaded during source verification.
