"""Final reproducibility, provenance, result and report-claims audit."""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audio_wm_eval.common import ROOT, configure_runtime, sha256, write_json
configure_runtime()
import numpy as np
import pandas as pd
import yaml
from audio_wm_eval.experiment import read_manifest, validate_rows
from audio_wm_eval.reporting import load_raw, aggregate_results

def main():
    checks=[]
    def check(name,passed,details=''):
        checks.append({'name':name,'passed':bool(passed),'details':details})
    clips=read_manifest(ROOT/'data/manifests/evaluation_manifest.csv')
    conditions=yaml.safe_load((ROOT/'configs/attacks.yaml').read_text())['conditions']
    with (ROOT/'outputs/raw_results.csv').open() as f:
        rows=list(csv.DictReader(f))
    grid=validate_rows(rows,[c['clip_id'] for c in clips],[c['id'] for c in conditions])
    check('complete_full_grid',grid['actual_rows']==624 and grid['failed_rows']==0,grid)
    check('24_clips_12_speakers',len(clips)==24 and len({c['speaker_id'] for c in clips})==12)
    check('all_processed_hashes',all(sha256(ROOT/c['processed_path'])==c['processed_sha256'] for c in clips))
    models=json.loads((ROOT/'outputs/model_provenance.json').read_text())
    check('official_checkpoint_hashes',all(sha256(ROOT/models[k]['path'])==models[k]['sha256'] for k in ('generator','detector')))
    check('official_checkpoint_urls',all(models[k]['url'].startswith('https://huggingface.co/facebook/audioseal/resolve/') for k in ('generator','detector')))
    env=json.loads((ROOT/'outputs/environment.json').read_text())
    check('dependency_lock_unchanged',sha256(ROOT/'requirements-lock.txt')==env['lock_sha256'])
    check('official_audioseal_package',env['packages']['audioseal']=='0.2.0')
    full_manifest=json.loads((ROOT/'outputs/run_manifest.json').read_text())
    check('run_manifest_matches_data',full_manifest['manifest_sha256']==sha256(ROOT/'data/manifests/evaluation_manifest.csv'))
    raw=load_raw(ROOT/'outputs/raw_results.csv')
    recorded=pd.read_csv(ROOT/'outputs/summary_results.csv')
    computed=aggregate_results(raw,resamples=2000)
    pd.testing.assert_frame_equal(recorded,computed,check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-12)
    check('summaries_recomputed_from_raw',True,'Every summary cell checked against fresh 2000-resample computation')
    comparison=json.loads((ROOT/'outputs/reproducibility/comparison.json').read_text())
    check('independent_process_replay',comparison['status']=='passed' and comparison['rows_compared']==52,comparison)
    with (ROOT/'outputs/pilot/raw_results.csv').open() as f:
        pilot={(r['clip_id'],r['condition'],r['branch']):r for r in csv.DictReader(f)}
    full={(r['clip_id'],r['condition'],r['branch']):r for r in rows}
    check('pilot_and_full_sample_hashes',all(pilot[k]['output_sha256']==full[k]['output_sha256'] for k in pilot))
    check('pilot_and_full_scientific_endpoints',all(pilot[k][m]==full[k][m] for k in pilot for m in ('detection_score','detected','recovered_message','bit_error_rate')))
    check('attack_target_snr',all(abs(json.loads(r['attack_metadata'])['noise_snr_db']-json.loads(r['attack_setting'])['snr_db'])<.05 for r in rows if r['attack_family']=='noise'))
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    forbidden=[p for p in tracked if re.search(r'(^\.venv/|^\.cache/|^\.python/|^\.tools/|^checkpoints/|^data/(raw|processed)/|\.(wav|mp3|flac|pth|pt)$)',p)]
    check('no_binary_data_or_caches_in_git',not forbidden,forbidden)
    tests=json.loads((ROOT/'outputs/final_validation.json').read_text())
    check('full_test_suite',tests['status']=='passed',tests)
    technical=(ROOT/'reports/technical_report.md').read_text()
    panel=(ROOT/'reports/panel_summary.md').read_text()
    check('panel_one_page_word_budget',len(panel.split())<=550,{'words':len(panel.split()),'limit':550})
    check('report_scope_disclosed',all(s in technical.lower() for s in ('preliminary','24','12','false positives','no human listening','speaker','2,000','16-bit')))
    check('report_failures_disclosed','setuptools' in technical and 'STATUS.md' in technical)
    check('required_reports_figures',all((ROOT/'outputs/figures'/f).is_file() for f in ('detection_rates.png','message_recovery.png','score_distributions.png','quality_vs_detection.png')))
    report={'status':'passed' if all(c['passed'] for c in checks) else 'failed',
            'checks':checks,'raw_sha256':sha256(ROOT/'outputs/raw_results.csv'),
            'technical_report_sha256':sha256(ROOT/'reports/technical_report.md'),
            'panel_summary_sha256':sha256(ROOT/'reports/panel_summary.md'),
            'limits':['One pretrained checkpoint pair; no training','24 audiobook speech clips; 12 dependent speaker clusters',
                      'Digital non-adaptive transformations only','No listening study; MP3 temporal alignment not verified',
                      'Replay establishes within-host repeatability for two clips, not bitwise cross-platform reproducibility']}
    write_json(ROOT/'outputs/claims_audit.json',report)
    print(json.dumps({'status':report['status'],'checks':len(checks),'failed':[c for c in checks if not c['passed']]},indent=2))
    if report['status']!='passed': raise SystemExit(1)

if __name__=='__main__': main()
