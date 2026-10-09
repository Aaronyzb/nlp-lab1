"""Produce numerical tables, confidence intervals and error evidence from saved predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from common import ROOT, LABELS, load_data, metrics, write_json

NAMES=['bow_binary','bow_frequency','glove','w2v_ag','w2v_nyt','bert']
DISPLAY=['Binary BoW','Frequency BoW','GloVe 100d','Word2Vec AG','Word2Vec NYT','BERT']

def fast_scores(y,p):
    matrix=np.bincount(3*y+p,minlength=9).reshape(3,3)
    denom=matrix.sum(0)+matrix.sum(1)
    f1=np.divide(2*np.diag(matrix),denom,out=np.zeros(3,dtype=float),where=denom!=0)
    return np.trace(matrix)/matrix.sum(),f1.mean()

def main():
    frame,y_all,split=load_data()
    y=y_all[split['test']]
    results={name:json.loads((ROOT/'results'/f'{name}.json').read_text(encoding='utf-8')) for name in NAMES}
    predictions={}
    for name in NAMES:
        table=pd.read_csv(ROOT/'results'/f'{name}_predictions.csv')
        np.testing.assert_array_equal(table.row_id,split['test'])
        predictions[name]=table.predicted_label.map(dict(zip(LABELS,range(3)))).to_numpy()
        assert metrics(y,predictions[name])==results[name]['test']
    majority=int(np.bincount(y_all[split['train']]).argmax())
    baseline=metrics(y,np.full(len(y),majority))
    rng=np.random.default_rng(123)
    strata=[np.flatnonzero(y==label) for label in range(3)]
    bootstrap={name:[] for name in NAMES}
    for _ in range(2000):
        ix=np.concatenate([rng.choice(group,size=len(group),replace=True) for group in strata])
        for name in NAMES: bootstrap[name].append(fast_scores(y[ix],predictions[name][ix]))
    intervals={name: {'accuracy_95ci':np.quantile(np.array(values)[:,0],[.025,.975]).tolist(),
                      'macro_f1_95ci':np.quantile(np.array(values)[:,1],[.025,.975]).tolist()}
               for name,values in bootstrap.items()}
    difference=np.array(bootstrap['bert'])-np.array(bootstrap['bow_binary'])
    paired={'accuracy_difference_bert_minus_binary_95ci':np.quantile(difference[:,0],[.025,.975]).tolist(),
            'macro_f1_difference_bert_minus_binary_95ci':np.quantile(difference[:,1],[.025,.975]).tolist()}
    summary=[]
    for name,display in zip(NAMES,DISPLAY):
        result=results[name]
        summary.append({'method':display, 'name':name, 'accuracy':result['test']['accuracy'],
                        'macro_f1':result['test']['macro_f1'],'elapsed_seconds':result['elapsed_seconds'],
                        **intervals[name]})
    pd.DataFrame(summary).to_csv(ROOT/'results'/'summary.csv',index=False)
    write_json(ROOT/'results'/'statistical_analysis.json', {'baseline':baseline,'confidence_intervals':intervals,
        'paired_bert_binary':paired,'bootstrap_replicates':2000,'bootstrap_seed':123,
        'method':'stratified paired bootstrap over test rows, percentile CI; conditional on fixed test class counts'})
    errors=[]
    for name in NAMES:
        for pos in np.flatnonzero(y!=predictions[name]):
            index=int(split['test'][pos])
            errors.append({'model':name,'row_id':index,'true':LABELS[y[pos]],'predicted':LABELS[predictions[name][pos]],
                           'text_excerpt':frame.text.iloc[index][:700]})
    pd.DataFrame(errors).to_csv(ROOT/'results'/'error_analysis.csv',index=False)
    import joblib
    saved=joblib.load(ROOT/'models'/'bow_binary_lr.joblib')
    words=saved['transform']['vectorizer'].get_feature_names_out()
    top={label:words[np.argsort(saved['classifier'].coef_[i])[-12:][::-1]].tolist() for i,label in enumerate(LABELS)}
    write_json(ROOT/'results'/'top_binary_features.json',top)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)

if __name__=='__main__': main()
