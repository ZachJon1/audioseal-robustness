import copy
import pytest
from audio_wm_eval.experiment import FIELDS, validate_rows, run_experiment

def valid_rows():
    rows=[]
    for branch in ('positive','negative'):
        positive=branch=='positive'
        row=dict.fromkeys(FIELDS)
        row.update(clip_id='clip',condition='clean',branch=branch,is_watermarked=int(positive),
                   attack_family='clean',status='ok',error_message='',
                   detection_score=float(positive),mean_frame_probability=float(positive),
                   detected=int(positive),detector_threshold=.5,output_peak=.5,
                   clipping_fraction=0.,prelimit_clipping_fraction=0.,num_samples=16000,
                   output_duration_seconds=1.,expected_message='0'*16,recovered_message='0'*16,
                   attack_runtime_ms=0.,detect_runtime_ms=1.)
        if positive:
            row.update(bit_errors=0,bit_accuracy=1.,bit_error_rate=0.,exact_message_recovery=1)
        rows.append(row)
    return rows

def test_complete_paired_rows_and_failures():
    rows=valid_rows()
    assert validate_rows(rows,['clip'],['clean'])['actual_rows']==2
    rows[0]['status']='failed'
    rows[0]['error_message']='Recorded intentional synthetic failure'
    assert validate_rows(rows,['clip'],['clean'])['failed_rows']==1

@pytest.mark.parametrize('problem',['missing','duplicate','field','threshold','message','negative_message',
                                    'negative_metric','branch','duration','unaligned','failed_reason','runtime'])
def test_invalid_results_are_rejected(problem):
    rows=copy.deepcopy(valid_rows())
    if problem=='missing': rows.pop()
    elif problem=='duplicate': rows.append(rows[0])
    elif problem=='field': del rows[0]['seed']
    elif problem=='threshold': rows[0]['detected']=0
    elif problem=='message': rows[0]['bit_error_rate']=.5
    elif problem=='negative_message': rows[1]['recovered_message']='bad'
    elif problem=='negative_metric': rows[1]['bit_accuracy']=1.
    elif problem=='branch': rows[1]['is_watermarked']=1
    elif problem=='duration': rows[0]['output_duration_seconds']=2
    elif problem=='unaligned': rows[0].update(attack_family='crop',quality_snr_db=0.)
    elif problem=='failed_reason': rows[0].update(status='failed',error_message='')
    elif problem=='runtime': rows[0]['detect_runtime_ms']=-1
    with pytest.raises(ValueError): validate_rows(rows,['clip'],['clean'])

def test_run_refuses_existing_raw_before_loading_models(tmp_path):
    path=tmp_path/'raw_results.csv'
    path.write_text('preserve me')
    with pytest.raises(FileExistsError): run_experiment(tmp_path)
    assert path.read_text()=='preserve me'
