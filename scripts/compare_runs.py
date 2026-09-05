"""Compare independent-process scientific outputs; wall-clock times are variable."""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from audio_wm_eval.common import write_json, sha256

EXCLUDED = {'run_id','git_commit','code_sha256','embed_runtime_ms','attack_runtime_ms','detect_runtime_ms'}
NUMERIC = {'detection_score','mean_frame_probability','watermark_snr_db','watermark_stoi','stoi','quality_snr_db'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    def read(path):
        with path.open() as f:
            return {(r['clip_id'],r['condition'],r['branch']):r for r in csv.DictReader(f)}
    ref,cand=read(args.reference),read(args.candidate)
    failures=[]
    max_error=0.
    if set(ref)!=set(cand):
        failures.append('Row keys differ')
    for key in sorted(set(ref)&set(cand)):
        for field,left in ref[key].items():
            if field in EXCLUDED:
                continue
            right=cand[key][field]
            if left==right:
                continue
            if field in NUMERIC and left!='NA' and right!='NA':
                a,b=float(left),float(right)
                if np.isclose(a,b,rtol=1e-7,atol=1e-7):
                    max_error=max(max_error,abs(a-b))
                    continue
            failures.append({'key':key,'field':field,'reference':left,'candidate':right})
    result={'status':'passed' if not failures else 'failed','rows_compared':len(set(ref)&set(cand)),
            'reference_sha256':sha256(args.reference),'candidate_sha256':sha256(args.candidate),
            'excluded_fields':sorted(EXCLUDED),'numeric_tolerance':{'rtol':1e-7,'atol':1e-7},
            'max_tolerated_difference':max_error,'failures':failures}
    write_json(args.output,result)
    print(result)
    if failures:
        raise SystemExit(1)

if __name__=='__main__':
    main()
