import hashlib
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from audio_wm_eval.common import message_for, stable_seed
from audio_wm_eval.audio_io import save_audio

def test_message_golden_and_independent_process():
    assert message_for('123-45-0000')=='1001111100000000'
    code="from audio_wm_eval.common import message_for,stable_seed; print(message_for('123-45-0000'),stable_seed(20260905,'clip','crop_10','attack'))"
    env={**os.environ,'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')}
    a=subprocess.check_output([sys.executable,'-c',code],env=env)
    env['PYTHONHASHSEED']='4321'
    b=subprocess.check_output([sys.executable,'-c',code],env=env)
    assert a==b
    assert stable_seed(20260905,'a')!=stable_seed(20260905,'b')

def test_wav_bytes_reproducible(tmp_path):
    x=np.random.default_rng(20260905).uniform(-.9,.9,16000).astype(np.float32)
    a,b=tmp_path/'a.wav',tmp_path/'b.wav'
    save_audio(a,x)
    save_audio(b,x)
    assert hashlib.sha256(a.read_bytes()).digest()==hashlib.sha256(b.read_bytes()).digest()
