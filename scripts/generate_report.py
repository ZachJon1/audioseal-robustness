#!/usr/bin/env python3
"""Create all aggregate tables, figures, and reports from raw per-clip results."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_wm_eval.reporting import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="outputs/raw_results.csv")
    parser.add_argument("--output-dir", default="outputs/summary")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    metadata = generate_report(args.raw, args.output_dir, args.reports_dir, args.resamples)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
