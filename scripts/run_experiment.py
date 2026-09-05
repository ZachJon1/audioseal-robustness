import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from audio_wm_eval.experiment import run_experiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run official pretrained AudioSeal paired evaluation')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--save-audio', choices=['none','first','all'], default='first')
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    print(run_experiment(args.output_dir, args.limit, args.save_audio, args.config))
