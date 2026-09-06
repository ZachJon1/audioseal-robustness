"""Run the entire suite and save machine-readable test evidence without overwrite."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    xml_path = args.output_dir/'test_suite.xml'
    log_path = args.output_dir/'test_suite.log'
    result_path = args.output_dir/'final_validation.json'
    for path in (xml_path, log_path, result_path):
        if path.exists():
            raise FileExistsError(f'Refusing to replace {path}; choose a fresh --output-dir')
    command = [sys.executable, '-m', 'pytest', '-q', f'--junitxml={xml_path}']
    with log_path.open('x') as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    suites = list(ET.parse(xml_path).getroot().iter('testsuite')) if xml_path.exists() else []
    counts = {key:sum(int(s.get(key, '0')) for s in suites) for key in ('tests','failures','errors','skipped')}
    result = {'status':'passed' if process.returncode == 0 and counts['tests'] else 'failed',
              'command':command,'returncode':process.returncode,
              'tests_passed':counts['tests']-counts['failures']-counts['errors']-counts['skipped'],
              'tests_failed':counts['failures']+counts['errors'],'tests_skipped':counts['skipped'],
              'log':str(log_path),'junit':str(xml_path)}
    with result_path.open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(process.returncode or (0 if result['status']=='passed' else 1))

if __name__ == '__main__':
    main()
