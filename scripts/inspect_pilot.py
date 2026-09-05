"""Numerical and waveform inspection of every saved two-clip pilot output."""
import csv
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from audio_wm_eval.common import ROOT, configure_runtime, write_json, sha256
configure_runtime()
from audio_wm_eval.experiment import read_manifest, validate_rows
from audio_wm_eval.audio_io import load_audio
import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    directory = ROOT/'outputs/pilot'
    with (directory/'raw_results.csv').open() as f:
        rows = list(csv.DictReader(f))
    clips = read_manifest(ROOT/'data/manifests/evaluation_manifest.csv')[:2]
    conditions = yaml.safe_load((ROOT/'configs/attacks.yaml').read_text())['conditions']
    audit = validate_rows(rows, [c['clip_id'] for c in clips], [c['id'] for c in conditions])
    assert audit['failed_rows'] == 0, 'Inspect failed pilot rows before full run'
    assert len({c['speaker_id'] for c in clips}) == 2
    audit['audio_checks'] = []
    for row in rows:
        path = directory/'audio_examples'/f"{row['clip_id']}_{row['condition']}_{row['branch']}.wav"
        audio, sr = load_audio(path)
        rms = float(np.sqrt(np.mean(audio.astype(float)**2)))
        assert len(audio) == int(row['num_samples']) and rms > 1e-6
        family = row['attack_family']
        meta = json.loads(row['attack_metadata'])
        if family == 'noise':
            assert abs(meta['noise_snr_db_pre_clipping'] - meta['snr_db']) < 1e-8
            assert abs(meta['noise_snr_db'] - meta['snr_db']) < .05, 'Limiting invalidates target noise SNR'
        if family == 'crop':
            assert abs(meta['removed_fraction_actual']-meta['fraction']) < 1/int(meta['input_samples'])
        if family == 'stretch':
            assert abs(len(audio)-meta['input_samples']/meta['rate']) <= 1
        if row['condition'] == 'clean':
            assert int(row['detected']) == int(row['is_watermarked']), 'Unexpected clean pilot detection needs review'
            if row['branch'] == 'positive':
                assert float(row['bit_error_rate']) == 0, 'Unexpected clean pilot message error needs review'
        audit['audio_checks'].append({'file':str(path.relative_to(ROOT)), 'rms':rms,
                                      'peak':float(np.max(np.abs(audio))), 'samples':len(audio)})
    selected = ['clean','mp3_64','noise_20','resample_8000','pitch_plus2','stretch_09','crop_25']
    fig, axes = plt.subplots(7, 2, figsize=(12, 14), constrained_layout=True)
    clip = clips[0]['clip_id']
    for axs, condition in zip(axes, selected):
        x, sr = load_audio(directory/'audio_examples'/f'{clip}_{condition}_positive.wav')
        axs[0].plot(np.arange(len(x))/sr, x, linewidth=.25)
        axs[0].set(title=condition+' waveform', ylabel='Amplitude', ylim=(-1,1))
        axs[1].specgram(x, NFFT=512, Fs=sr, noverlap=384, cmap='magma', vmin=-100, vmax=-30)
        axs[1].set(title=condition+' spectrogram', ylabel='Hz')
    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')
    fig.suptitle(f'Pilot waveform/spectrum inspection: {clip}; objective inspection only')
    figure = directory/'inspection.png'
    if figure.exists():
        raise FileExistsError(figure)
    fig.savefig(figure, dpi=120)
    plt.close(fig)
    validation = json.loads((directory/'validation.json').read_text())
    audit.update(raw_sha256=sha256(directory/'raw_results.csv'),
                 listening_status='No human listening performed; decoded samples, waveforms, spectra and numerical metrics inspected.',
                 estimated_full_runtime_seconds=validation['total_runtime_seconds']*12,
                 estimated_full_examples_bytes=sum(p.stat().st_size for p in (directory/'audio_examples').glob('*.wav'))//2,
                 status='passed')
    write_json(directory/'inspection.json', audit)
    print(json.dumps({k:v for k,v in audit.items() if k != 'audio_checks'}, indent=2))

if __name__ == '__main__':
    main()
