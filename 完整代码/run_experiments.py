"""One command for resource download, all six experiments, checks and report."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--classical-python', default=sys.executable)
    parser.add_argument('--bert-python', default=sys.executable)
    parser.add_argument('--skip-download', action='store_true')
    args=parser.parse_args()
    env={**os.environ,'PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8'}
    (ROOT/'logs').mkdir(exist_ok=True)
    def run(python, arguments, log):
        print('RUN',arguments,flush=True)
        with (ROOT/'logs'/log).open('w',encoding='utf-8') as stream:
            subprocess.run([python,'-u',*arguments],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if not args.skip_download: run(args.bert_python,['src/download_resources.py','all'],'download_resources.log')
    run(args.classical_python,['src/common.py'],'data_audit.log')
    run(args.classical_python,['src/train_classical.py','--task','all'],'train_classical.log')
    run(args.bert_python,['src/train_bert.py'],'train_bert.log')
    run(args.classical_python,['-m','unittest','discover','-s','tests','-v'],'tests.log')
    run(args.bert_python,['src/analyze_results.py'],'analysis.log')
    run(args.classical_python,['src/report_evidence.py'],'report_evidence.log')
    run(args.bert_python,['src/report_charts.py'],'report_charts.log')
    run(args.bert_python,['src/report_log_images.py'],'report_log_images.log')
    run(args.bert_python,['src/package_submission.py','--provenance-only'],'provenance.log')
    run(args.bert_python,['src/build_report.py'],'report.log')
    print('Done. See report/ and results/.',flush=True)

if __name__=='__main__': main()
