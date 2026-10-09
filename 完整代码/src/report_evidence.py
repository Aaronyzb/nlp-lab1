"""Recompute saved test metrics and create evidence pages for the report."""
from pathlib import Path
import ast
import html
import json
import subprocess
import sys
import numpy as np
import pandas as pd
from common import ROOT, LABELS, load_data, metrics, write_json
from analyze_results import NAMES

DISPLAY=['Binary BoW','Word Frequency','GloVe 100d','Word2Vec External','Word2Vec Target','BERT']

def source_lines(path, start, end):
    lines=(ROOT/path).read_text(encoding='utf-8').splitlines()
    return '\n'.join(f'{i:3d}  {lines[i-1]}' for i in range(start,end+1))

def function_lines(path,name):
    text=(ROOT/path).read_text(encoding='utf-8')
    node=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    return node.lineno,node.end_lineno

def main():
    assets=ROOT/'report'/'assets'; assets.mkdir(parents=True,exist_ok=True)
    pages=ROOT/'tmp'/'evidence_pages'; pages.mkdir(parents=True,exist_ok=True)
    frame,y,split=load_data()
    evidence=[]
    output=['SAVED TEST PREDICTIONS - METRIC CHECK',f'Test samples: {len(split["test"])}',
            '',f'{"Model":<22} {"Accuracy":>10} {"Macro-F1":>10} {"Errors":>8}']
    for name,display in zip(NAMES,DISPLAY):
        table=pd.read_csv(ROOT/'results'/f'{name}_predictions.csv')
        np.testing.assert_array_equal(table.row_id,split['test'])
        pred=table.predicted_label.map(dict(zip(LABELS,range(3)))).to_numpy()
        result=metrics(y[split['test']],pred)
        saved=json.loads((ROOT/'results'/f'{name}.json').read_text(encoding='utf-8'))
        assert result==saved['test']
        errors=int(np.sum(pred!=y[split['test']]))
        output.append(f'{display:<22} {result["accuracy"]:>10.6f} {result["macro_f1"]:>10.6f} {errors:>8}')
        evidence.append({'model':name,'test_rows':len(table),'metrics_match':True,'errors':errors})
    output+=['','PASS: all six metrics match the saved results.',
             'PASS: all models use identical test row IDs.']
    result_text='\n'.join(output)
    (ROOT/'logs'/'report_metric_check.log').write_text(result_text,encoding='utf-8')
    print(result_text,flush=True)
    # This is a new verification run, separate from the saved training logs.
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],
                         cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    test_text=test.stdout+test.stderr
    (ROOT/'logs'/'report_test_check.log').write_text(test_text,encoding='utf-8')
    if test.returncode: raise RuntimeError(test_text)
    bert_log=(ROOT/'logs'/'train_bert.log').read_text(encoding='utf-8').splitlines()
    selected=[line for line in bert_log if line.startswith('Device ') or
              ('step 288/288' in line) or line.startswith("{'epoch':") or line.startswith('FINISHED BERT')]
    # Preserve log lines and numbers; wrapping happens only in the display.
    snippets={
        'metric_check':('逐篇测试预测的重新核算','logs/report_metric_check.log',result_text),
        'test_check':('本次正确性检查输出','logs/report_test_check.log',test_text.strip()),
        'bert_log':('保存的三轮训练日志节选','logs/train_bert.log','\n'.join(selected)),
    }
    start,end=function_lines(Path('src/common.py'),'metrics')
    snippets['metric_code']=('评价函数源代码',f'src/common.py:{start}-{end}',source_lines(Path('src/common.py'),start,end))
    start,end=function_lines(Path('src/train_classical.py'),'mean_embeddings')
    snippets['pooling_code']=('平均词向量源代码',f'src/train_classical.py:{start}-{end}',source_lines(Path('src/train_classical.py'),start,end))
    for name,(title,source,content) in snippets.items():
        page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{margin:0;background:#fff;color:#17232d;font-family:"Microsoft YaHei",sans-serif}}
main{{width:960px;box-sizing:border-box;padding:30px 34px}}h1{{font-size:23px;margin:0 0 10px;font-weight:600}}
.source{{font-size:15px;color:#536574;margin-bottom:22px}}pre{{font:19px/1.55 Consolas,monospace;white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:23px;background:#f2f5f8;border:1px solid #dbe3ea}}
.note{{font-size:14px;color:#657786;margin-top:16px}}</style>
<main><h1>{html.escape(title)}</h1><div class="source">来源 {html.escape(source)}</div>
<pre>{html.escape(content)}</pre><div class="note">显示的是实际文件内容，长行仅作折行处理。</div></main></html>'''
        (pages/f'{name}.html').write_text(page,encoding='utf-8')
    write_json(ROOT/'results'/'report_evidence.json',{'metric_checks':evidence,'tests_return_code':test.returncode,
        'evidence_pages':{name:{'source':source,'title':title} for name,(title,source,_) in snippets.items()}})

if __name__=='__main__': main()
