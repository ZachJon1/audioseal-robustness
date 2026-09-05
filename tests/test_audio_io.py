import numpy as np
import pytest
from scipy.io import wavfile
from audio_wm_eval.audio_io import load_audio, save_audio, validate_audio

@pytest.mark.parametrize('audio,sr', [([],16000), ([[0.,0.]],16000), ([np.nan],16000),
                                    ([np.inf],16000), ([1.01],16000), ([0.],8000)])
def test_invalid_shape_range_rate(audio,sr):
    with pytest.raises(ValueError):
        validate_audio(audio,sr)

def test_roundtrip_and_overwrite(tmp_path):
    audio=np.array([-.99,0,.7],np.float32)
    path=tmp_path/'audio.wav'
    save_audio(path,audio)
    restored,sr=load_audio(path)
    np.testing.assert_array_equal(audio,restored)
    assert restored.dtype==np.float32 and sr==16000
    with pytest.raises(FileExistsError):
        save_audio(path,audio)

def test_noncanonical_audio_needs_explicit_conversion(tmp_path):
    path=tmp_path/'stereo.wav'
    wavfile.write(path,8000,np.zeros((800,2),np.float32))
    with pytest.raises(ValueError):
        load_audio(path)
    audio,sr=load_audio(path,convert=True)
    assert audio.shape==(1600,) and sr==16000
