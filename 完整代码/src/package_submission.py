"""Verify final artifacts, record provenance and package the GitHub submission files."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys
import zipfile
from common import ROOT, write_json

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def provenance():
    sources={}
    for path in [ROOT/'resources'/'glove.6B.100d.txt',ROOT/'resources'/'bert-base-uncased'/'model.safetensors']:
        sources[path.name]={'bytes':path.stat().st_size,'sha256':digest(path)}
    sources['glove_source']='https://nlp.stanford.edu/data/glove.6B.zip'
    sources['bert_source']=json.loads((ROOT/'resources'/'bert_source.json').read_text(encoding='utf-8'))
    sources['bert_environment']={'python':sys.version,'platform':platform.platform(),
        'versions':{package:version(package) for package in ['torch','transformers','huggingface-hub',
                   'numpy','pandas','scikit-learn','reportlab','matplotlib','PyMuPDF']}}
    sources['classical_environment']=json.loads((ROOT/'results'/'w2v_nyt.json').read_text(encoding='utf-8'))['metadata']['environment']
    write_json(ROOT/'results'/'resource_provenance.json',sources)
    print('Resource provenance recorded',flush=True)

def package():
    import fitz
    pdf=ROOT/'report'/'最新实验报告.pdf'
    if not pdf.exists():
        pdf=ROOT/'report'/'文本分类实验报告.pdf'
    with fitz.open(pdf) as doc:
        pages=len(doc)
        for page in doc:
            if not page.get_text().strip(): raise ValueError('Empty PDF page')
    checked=json.loads((ROOT/'results'/'verification.json').read_text(encoding='utf-8'))
    from analyze_results import NAMES
    if checked.get('completed_experiments') != NAMES: raise ValueError('Verification does not match the required six experiments')
    # The user-selected final PDF is copied verbatim; generated-report review
    # metadata applies only to the earlier generated report.
    write_json(ROOT/'results'/'submission_report.json',{
        'source':str(pdf.relative_to(ROOT)).replace('\\','/'),
        'sha256':digest(pdf),'pages':pages,'selection':'user-selected final report',
        'copied_without_modification':True})
    paths=[]
    for folder in ['src','tests','HW-1','results','logs']:
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and
                     '__pycache__' not in p.parts and p.name != '.DS_Store')
    paths.extend(p for p in (ROOT/'report'/'assets').rglob('*') if p.is_file())
    paths.extend(p for p in ROOT.iterdir() if p.is_file() and
                 (p.suffix in ['.py','.txt'] or p.name in ['README.md','.gitignore']))
    paths=sorted(set(paths))
    staging=ROOT/'提交材料'
    staging.mkdir(exist_ok=True)
    members={}
    for path in paths:
        relative=Path('完整代码')/path.relative_to(ROOT)
        destination=staging/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        members[relative.as_posix()]=destination
    for relative,source in [('实验报告.pdf',pdf),('代码运行说明.md',ROOT/'代码运行说明.md')]:
        destination=staging/relative
        shutil.copy2(source,destination)
        members[relative]=destination
    manifest={name:{'bytes':path.stat().st_size,'sha256':digest(path)} for name,path in members.items()}
    manifest_path=staging/'完整代码'/'SUBMISSION_MANIFEST.json'
    write_json(manifest_path,manifest)
    write_json(ROOT/'SUBMISSION_MANIFEST.json',manifest)
    members['完整代码/SUBMISSION_MANIFEST.json']=manifest_path
    if {p.name for p in staging.iterdir()} != {'实验报告.pdf','完整代码','代码运行说明.md'}:
        raise ValueError('Submission folder must contain exactly the three requested top-level items')
    target=ROOT/'submission.zip'
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in sorted(members.items()): archive.write(path,name)
    with zipfile.ZipFile(target) as archive:
        failed=archive.testzip()
        if failed: raise ValueError('Corrupted archive member '+failed)
        assert len(archive.namelist())==len(members)
        assert {name.split('/')[0] for name in archive.namelist()}=={'实验报告.pdf','完整代码','代码运行说明.md'}
    assert digest(staging/'实验报告.pdf')==digest(pdf)
    print(f'Packaged {len(members)} files; {target.stat().st_size/1e6:.2f} MB; PDF {pages} pages; three top-level items',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--provenance-only',action='store_true')
    args=parser.parse_args()
    provenance()
    if not args.provenance_only: package()
