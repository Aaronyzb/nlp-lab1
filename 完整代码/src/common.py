"""Shared split, tokenization, evaluation and audit utilities."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
LABELS = ['business', 'politics', 'sports']
TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+(?:\.[0-9]+)?")

def tokenize(text):
    """Lowercase English words, contractions and numbers; no external NLTK download."""
    return TOKEN.findall(str(text).lower())

def stable_hash(word):
    return int.from_bytes(hashlib.md5(word.encode('utf-8')).digest()[:4], 'little')

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')

def load_data():
    frame = pd.read_csv(ROOT/'HW-1'/'nyt.csv', keep_default_na=False)
    if list(frame.columns) != ['text', 'label'] or set(frame.label) != set(LABELS):
        raise ValueError('Unexpected NYT schema or labels')
    if not frame.text.str.strip().all(): raise ValueError('Empty document')
    y = frame.label.map({label: i for i, label in enumerate(LABELS)}).to_numpy()
    permutation = np.random.default_rng(SEED).permutation(len(frame))
    n_train, n_train_val = int(.8*len(frame)), int(.9*len(frame))
    indices = dict(train=permutation[:n_train], validation=permutation[n_train:n_train_val],
                   test=permutation[n_train_val:])
    split_path = ROOT/'results'/'split_indices.json'
    serial = {key: values.tolist() for key, values in indices.items()}
    if split_path.exists():
        if json.loads(split_path.read_text(encoding='utf-8')) != serial:
            raise ValueError('Existing split differs; do not silently change the split')
    else: write_json(split_path, serial)
    return frame, y, indices

def metrics(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=range(3), average='macro', zero_division=0)),
            'classification_report': classification_report(y, pred, labels=range(3),
                  target_names=LABELS, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(y, pred, labels=range(3)).tolist()}

def save_predictions(name, indices, y, pred, frame):
    pd.DataFrame({'row_id': indices, 'true_label': [LABELS[i] for i in y],
                  'predicted_label': [LABELS[i] for i in pred]}).to_csv(
                      ROOT/'results'/f'{name}_predictions.csv', index=False)
    train_texts = set(frame.text.iloc[load_data()[2]['train']])
    independent = np.array([text not in train_texts for text in frame.text.iloc[indices]])
    return {'test_without_exact_train_duplicates_n': int(independent.sum()),
            'test_without_exact_train_duplicates': metrics(y[independent], pred[independent])}

def audit_data():
    frame, y, split = load_data()
    ag = pd.read_csv(ROOT/'HW-1'/'ag.csv', keep_default_na=False)
    texts = {key: set(frame.text.iloc[ix]) for key, ix in split.items()}
    token_lengths = np.array([len(tokenize(t)) for t in frame.text])
    audit = {'seed': SEED, 'split_algorithm': 'numpy default_rng(42) permutation; floor(0.8N), floor(0.9N)',
             'labels': LABELS, 'nyt_rows': len(frame), 'ag_rows': len(ag),
             'nyt_duplicate_rows': int(frame.text.duplicated().sum()),
             'ag_duplicate_rows': int(ag.text.duplicated().sum()),
             'nyt_conflicting_label_texts': int((frame.groupby('text').label.nunique()>1).sum()),
             'split': {k: {'n': len(ix), 'class_counts': frame.label.iloc[ix].value_counts().to_dict()}
                       for k, ix in split.items()},
             'exact_duplicate_unique_text_overlap': {
                 f'{a}_{b}': len(texts[a]&texts[b])
                 for a,b in [('train','validation'),('train','test'),('validation','test')]},
             'ag_nyt_exact_text_overlap': len(set(ag.text)&set(frame.text)),
             'token_lengths': {'mean': float(token_lengths.mean()), 'median': float(np.median(token_lengths)),
                               'p90': float(np.quantile(token_lengths,.9)), 'max': int(token_lengths.max())},
             'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (ROOT/'HW-1').glob('*.csv')}}
    write_json(ROOT/'results'/'data_audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit

if __name__ == '__main__': audit_data()
