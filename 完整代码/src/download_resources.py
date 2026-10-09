"""Download GloVe 100d and the pinned BERT checkpoint."""
from pathlib import Path
import argparse
import json
import os
import time
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
BERT_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'

def download_glove():
    folder = ROOT / 'resources'
    folder.mkdir(exist_ok=True)
    target = folder / 'glove.6B.100d.txt'
    if target.exists():
        print('GloVe already downloaded', flush=True)
        return
    archive = folder / 'glove.6B.zip'
    url = 'https://nlp.stanford.edu/data/glove.6B.zip'
    if not archive.exists():
        partial = archive.with_suffix('.zip.part')
        for attempt in range(5):
            try:
                start = partial.stat().st_size if partial.exists() else 0
                with requests.get(url, headers={'Range': f'bytes={start}-'} if start else {},
                                  stream=True, timeout=(30, 120)) as response:
                    response.raise_for_status()
                    mode = 'ab' if start and response.status_code == 206 else 'wb'
                    if mode == 'wb': start = 0
                    total = start + int(response.headers.get('Content-Length', 0))
                    last = time.time()
                    with partial.open(mode) as stream:
                        for chunk in response.iter_content(1024 * 1024):
                            stream.write(chunk)
                            start += len(chunk)
                            if time.time() - last > 20:
                                print(f'GloVe download {start/1e6:.1f}/{total/1e6:.1f} MB', flush=True)
                                last = time.time()
                partial.replace(archive)
                break
            except requests.RequestException as exc:
                print(f'Download retry {attempt+1}: {exc}', flush=True)
                if attempt == 4: raise
    with zipfile.ZipFile(archive) as z:
        # Reading the whole selected member also validates the ZIP CRC.
        with z.open('glove.6B.100d.txt') as source, target.open('wb') as dest:
            import shutil
            shutil.copyfileobj(source, dest, 1024 * 1024)
    print('Extracted official glove.6B.100d.txt', flush=True)

def download_bert():
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    model = 'google-bert/bert-base-uncased'
    revision = BERT_REVISION
    source_record = ROOT/'resources'/'bert_source.json'
    if source_record.exists() and json.loads(source_record.read_text(encoding='utf-8'))['revision'] != revision:
        raise ValueError('Existing BERT resource revision differs from the pinned experiment revision')
    folder = ROOT/'resources'/'bert-base-uncased'
    folder.mkdir(parents=True, exist_ok=True)
    for name in ['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.txt']:
        target = folder/name
        if target.exists(): continue
        url = f'https://huggingface.co/{model}/resolve/{revision}/{name}'
        with requests.get(url, stream=True, timeout=(30,120)) as response:
            response.raise_for_status()
            total = int(response.headers.get('Content-Length',0))
            downloaded, last = 0, time.time()
            with target.with_suffix(target.suffix+'.part').open('wb') as stream:
                for chunk in response.iter_content(1024*1024):
                    stream.write(chunk); downloaded += len(chunk)
                    if time.time()-last > 20:
                        print(f'{name}: {downloaded/1e6:.1f}/{total/1e6:.1f} MB',flush=True); last=time.time()
            if total and downloaded != total: raise RuntimeError('Incomplete BERT resource')
        target.with_suffix(target.suffix+'.part').replace(target)
    (ROOT/'resources'/'bert_source.json').write_text(
        json.dumps({'model': model, 'revision': revision}, indent=2), encoding='utf-8')
    print(f'BERT downloaded at revision {revision}', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('resource', choices=['glove', 'bert', 'all'], default='all', nargs='?')
    args = parser.parse_args()
    if args.resource in ['glove', 'all']: download_glove()
    if args.resource in ['bert', 'all']: download_bert()
