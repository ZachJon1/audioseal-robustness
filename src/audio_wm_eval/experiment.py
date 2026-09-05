"""Paired inference, durable per-clip rows and explicit failure accounting."""
import csv
import hashlib
import json
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import yaml
from .common import ROOT, SEED, configure_runtime, stable_seed, message_for, sha256, write_json, code_hash
from .audio_io import load_audio, save_audio, validate_audio
from .audioseal_adapter import AudioSealAdapter
from .attacks import apply_attack
from .metrics import aligned_snr, clipping_fraction, compare_messages, stoi_score

FIELDS = '''run_id git_commit code_sha256 manifest_sha256 configuration_sha256 clip_id speaker_id source_split duration_seconds seed branch is_watermarked condition attack_family attack_setting detector_threshold frame_threshold detection_score mean_frame_probability detected expected_message recovered_message bit_errors bit_accuracy bit_error_rate exact_message_recovery watermark_snr_db watermark_stoi stoi pesq quality_snr_db quality_applicability output_duration_seconds num_samples output_peak clipping_fraction prelimit_clipping_fraction embed_runtime_ms attack_runtime_ms detect_runtime_ms status error_message attack_metadata output_sha256'''.split()

def read_manifest(path):
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or len({r['clip_id'] for r in rows}) != len(rows):
        raise ValueError('Manifest is empty or has duplicate clip IDs')
    for row in rows:
        if row['expected_message'] != message_for(row['clip_id'], int(row['seed'])):
            raise ValueError('Manifest message does not match independent seed derivation')
        if int(row['sample_rate']) != 16000 or not 2 <= float(row['duration_seconds']) <= 10:
            raise ValueError('Manifest sample rate/duration violates study scope')
    return rows

def validate_rows(rows, clip_ids, condition_ids):
    expected = {(c, a, b) for c in clip_ids for a in condition_ids for b in ('positive','negative')}
    keys = [(r['clip_id'], r['condition'], r['branch']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Missing, unexpected, or duplicate clip-condition-branch rows')
    for r in rows:
        if set(FIELDS) - set(r):
            raise ValueError('Incomplete result schema')
        if int(r['is_watermarked']) != int(r['branch'] == 'positive'):
            raise ValueError('Branch and watermark label disagree')
        if r['status'] not in ('ok', 'failed'):
            raise ValueError('Unknown result status')
        if r['status'] == 'failed':
            if not r['error_message']:
                raise ValueError('Failure needs an explicit error message')
            continue
        compare_messages(r['expected_message'], r['recovered_message'])
        for field in ('attack_runtime_ms', 'detect_runtime_ms'):
            if not np.isfinite(float(r[field])) or float(r[field]) < 0:
                raise ValueError('Runtime must be finite and nonnegative')
        for field in ['detection_score','mean_frame_probability','output_peak','clipping_fraction','prelimit_clipping_fraction']:
            if not 0 <= float(r[field]) <= 1:
                raise ValueError(f'Invalid {field}')
        if int(r['detected']) != int(float(r['detection_score']) > float(r['detector_threshold'])):
            raise ValueError('Thresholded detection is inconsistent')
        if int(r['num_samples']) < 1 or abs(float(r['output_duration_seconds']) - int(r['num_samples'])/16000) > 1e-9:
            raise ValueError('Invalid duration or output shape')
        if r['branch'] == 'positive':
            check = compare_messages(r['expected_message'], r['recovered_message'])
            for key in ['bit_errors','bit_accuracy','bit_error_rate','exact_message_recovery']:
                if float(r[key]) != float(check[key]):
                    raise ValueError('Message metric mismatch')
        else:
            if any(r[f] not in (None, '', 'NA') for f in ['bit_errors','bit_accuracy','bit_error_rate','exact_message_recovery','watermark_snr_db']):
                raise ValueError('Negative controls have no embedded-message recovery endpoint')
        if r['attack_family'] in ('mp3','pitch','stretch','crop'):
            if r['quality_snr_db'] not in (None, '', 'NA') or r['stoi'] not in (None, '', 'NA'):
                raise ValueError('Unaligned quality metric must be inapplicable')
    return {'expected_rows': len(expected), 'actual_rows': len(rows),
            'failed_rows': sum(r['status'] == 'failed' for r in rows), 'status': 'passed'}

def run_experiment(output_dir, limit=None, save_examples='first', config_path=None):
    output_dir = Path(output_dir)
    raw_path = output_dir / 'raw_results.csv'
    if raw_path.exists():
        raise FileExistsError(f'Refusing to overwrite {raw_path}; select a new output directory')
    config_path = Path(config_path or ROOT/'configs/experiment.yaml')
    config = yaml.safe_load(config_path.read_text())
    if config['seed'] != SEED or config['sample_rate'] != 16000:
        raise ValueError('Initial study uses fixed seed 20260905 and 16000 Hz')
    configure_runtime(config['seed'], config['torch_threads'])
    smoke = json.loads((ROOT/'outputs/smoke_test.json').read_text())
    if smoke['status'] != 'passed':
        raise RuntimeError('Official smoke gate has not passed')
    conditions = yaml.safe_load((ROOT/config['attacks_config']).read_text())['conditions']
    if len({c['id'] for c in conditions}) != len(conditions):
        raise ValueError('Duplicate attack IDs')
    manifest_path = ROOT/config['manifest']
    clips = read_manifest(manifest_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError('Limit must be positive')
        clips = clips[:limit]
    elif len(clips) != config['num_clips']:
        raise ValueError('Full manifest count does not match config')
    if limit is None:
        pilot = ROOT/'outputs/pilot/inspection.json'
        if not pilot.exists() or json.loads(pilot.read_text())['status'] != 'passed':
            raise RuntimeError('Full run requires inspected passing two-clip pilot')
    commit = subprocess.run(['git','rev-parse','HEAD'], cwd=ROOT, capture_output=True, text=True)
    if commit.returncode:
        raise RuntimeError('Commit project code before a reported experiment')
    git_commit = commit.stdout.strip()
    dirty = subprocess.run(['git','status','--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    config_sha = hashlib.sha256(config_path.read_bytes()+(ROOT/config['attacks_config']).read_bytes()).hexdigest()
    provenance = {'run_id': run_id, 'git_commit': git_commit, 'git_status': dirty,
                  'code_sha256': code_hash(), 'manifest_sha256': sha256(manifest_path),
                  'configuration_sha256': config_sha, 'config': config, 'conditions': conditions,
                  'clip_ids': [c['clip_id'] for c in clips], 'command': __import__('sys').argv,
                  'timing_definition': 'Wall time. Model loading and warmup separate. Embed measured once per clip and repeated in condition rows; do not sum duplicated embed times. Attack includes codec I/O and first library imports. Quality/file-write time only in total wall time.',
                  'missing_value': 'NA; see quality_applicability, branch and error_message'}
    write_json(output_dir/'run_manifest.json', provenance)
    rows = []
    started = time.perf_counter()
    adapter = None
    model_error = None
    try:
        adapter = AudioSealAdapter(config)
        warmup, _ = load_audio(ROOT/'data/raw/official_test.wav', convert=True)
        warm_y, _ = adapter.embed(warmup, message_for('official_test'))
        adapter.detect(warm_y)
        adapter.detect(warmup)
    except Exception:
        model_error = traceback.format_exc()
    # Rows are flushed individually before any aggregation; failures stay in place.
    with raw_path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        handle.flush()
        for index, clip in enumerate(clips):
            audio = marked = None
            input_error = model_error
            embed_error = None
            embed_ms = wm_snr = wm_stoi = None
            try:
                if model_error:
                    raise RuntimeError(model_error)
                if sha256(ROOT/clip['processed_path']) != clip['processed_sha256']:
                    raise ValueError('Processed audio hash mismatch against manifest')
                audio, sr = load_audio(ROOT/clip['processed_path'])
                if len(audio) != int(clip['num_samples']):
                    raise ValueError('Manifest sample count mismatch')
            except Exception:
                input_error = traceback.format_exc()
            if not input_error:
                try:
                    marked, embed_ms = adapter.embed(audio, clip['expected_message'])
                    wm_snr = aligned_snr(audio, marked)
                    wm_stoi = stoi_score(audio, marked, 16000)
                except Exception:
                    embed_error = traceback.format_exc()
            for condition in conditions:
                attack_seed = stable_seed(SEED, clip['clip_id'], condition['id'], 'attack')
                for branch in ('positive','negative'):
                    positive = branch == 'positive'
                    row = dict.fromkeys(FIELDS)
                    row.update({k: provenance[k] for k in ('run_id','git_commit','code_sha256','manifest_sha256','configuration_sha256')})
                    row.update(clip_id=clip['clip_id'], speaker_id=clip['speaker_id'], source_split=clip['source_split'],
                               duration_seconds=float(clip['duration_seconds']), seed=SEED, branch=branch,
                               is_watermarked=int(positive), condition=condition['id'], attack_family=condition['family'],
                               attack_setting=json.dumps(condition, sort_keys=True), detector_threshold=config['detector_threshold'],
                               frame_threshold=config['frame_threshold'], expected_message=clip['expected_message'],
                               status='failed', quality_applicability='NA: not evaluated',
                               watermark_snr_db=wm_snr if positive else None, watermark_stoi=wm_stoi if positive else None,
                               embed_runtime_ms=embed_ms if positive else None)
                    try:
                        failure = input_error or (embed_error if positive else None)
                        if failure:
                            raise RuntimeError(failure)
                        source = marked if positive else audio
                        attack_start = time.perf_counter()
                        changed, metadata = apply_attack(source, 16000, condition, attack_seed)
                        row['attack_runtime_ms'] = (time.perf_counter()-attack_start)*1000
                        validate_audio(changed)
                        row.update(adapter.detect(changed))
                        if positive:
                            comparison = compare_messages(clip['expected_message'], row['recovered_message'])
                            comparison['exact_message_recovery'] = int(comparison['exact_message_recovery'])
                            row.update(comparison)
                        aligned = metadata['sample_aligned'] and len(changed) == len(audio)
                        row.update(quality_snr_db=aligned_snr(audio, changed, aligned),
                                   stoi=stoi_score(audio, changed, 16000, aligned),
                                   quality_applicability='aligned_to_original' if aligned else 'NA: '+metadata.get('alignment_reason','incompatible length'),
                                   output_duration_seconds=len(changed)/16000, num_samples=len(changed),
                                   output_peak=float(np.max(np.abs(changed))), clipping_fraction=clipping_fraction(changed),
                                   prelimit_clipping_fraction=metadata['clipping_fraction'], attack_metadata=json.dumps(metadata, sort_keys=True),
                                   output_sha256=hashlib.sha256(changed.astype('<f4').tobytes()).hexdigest())
                        if save_examples == 'all' or (save_examples == 'first' and index == 0):
                            save_audio(output_dir/'audio_examples'/f"{clip['clip_id']}_{condition['id']}_{branch}.wav", changed)
                        row.update(status='ok', error_message='')
                    except Exception:
                        row.update(status='failed', error_message=traceback.format_exc())
                    writer.writerow({k: 'NA' if v is None else v for k,v in row.items()})
                    handle.flush()
                    rows.append(row)
            print(f"clip {index+1}/{len(clips)} {clip['clip_id']}: {sum(r['status']=='failed' for r in rows)} failed rows so far", flush=True)
    validation = validate_rows(rows, [c['clip_id'] for c in clips], [c['id'] for c in conditions])
    validation.update(total_runtime_seconds=time.perf_counter()-started,
                      model_load_runtime_ms=adapter.load_runtime_ms if adapter else None,
                      raw_sha256=sha256(raw_path))
    write_json(output_dir/'validation.json', validation)
    if validation['failed_rows']:
        raise RuntimeError(f"{validation['failed_rows']} failed evaluations recorded; inspect before proceeding")
    return validation
