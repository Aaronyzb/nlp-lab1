"""Tasks 1/2: fit on train, select on validation, evaluate test once per method."""
import argparse
import time
import warnings
import logging
import platform
import sys
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import ROOT, SEED, tokenize, stable_hash, load_data, metrics, write_json, save_predictions

CS = [.01, .1, 1., 10.]

def select_classifier(features, y, split):
    trials = []
    best, score = None, (-1., -1.)
    for weight in [None, 'balanced']:
        for c in CS:
            model = LogisticRegression(C=c, class_weight=weight, solver='lbfgs',
                                       max_iter=3000, tol=1e-4, random_state=SEED)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', ConvergenceWarning)
                model.fit(features[split['train']], y[split['train']])
            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                raise RuntimeError(f'LR did not converge: C={c}, class_weight={weight}')
            val = metrics(y[split['validation']], model.predict(features[split['validation']]))
            trial = {'C': c, 'class_weight': weight, 'validation_accuracy': val['accuracy'],
                     'validation_macro_f1': val['macro_f1'], 'n_iter': model.n_iter_.tolist()}
            trials.append(trial)
            print(trial, flush=True)
            rank = (val['macro_f1'], val['accuracy'])
            if rank > score: best, score = model, rank
    return best, trials

def finish(name, features, frame, y, split, metadata, started, transform=None):
    metadata['environment'] = {'python': sys.version, 'platform': platform.platform(),
        'versions': {package: version(package) for package in ['numpy','scipy','pandas','scikit-learn','joblib']}}
    model, trials = select_classifier(features, y, split)
    prediction = model.predict(features[split['test']])
    result = {'name': name, 'test': metrics(y[split['test']], prediction),
              'validation': metrics(y[split['validation']], model.predict(features[split['validation']])),
              'selected': {'C': model.C, 'class_weight': model.class_weight},
              'selection': 'maximize validation macro-F1, then accuracy; keep first tie',
              'trials': trials, 'metadata': metadata, 'elapsed_seconds': time.perf_counter()-started,
              'duplicate_sensitivity': save_predictions(name, split['test'], y[split['test']], prediction, frame)}
    write_json(ROOT/'results'/f'{name}.json', result)
    joblib.dump({'classifier': model, 'transform': transform}, ROOT/'models'/f'{name}_lr.joblib')
    print(f"FINISHED {name}: {result['test']['accuracy']:.6f}, {result['test']['macro_f1']:.6f}", flush=True)

def bow():
    frame, y, split = load_data()
    started = time.perf_counter()
    vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False,
                                 min_df=1, dtype=np.float64)
    vectorizer.fit(frame.text.iloc[split['train']])
    counts = vectorizer.transform(frame.text)
    common_preprocess_seconds = time.perf_counter()-started
    for name in ['bow_binary', 'bow_frequency']:
        start = time.perf_counter()
        if name == 'bow_binary':
            features = counts.copy()
            features.data[:] = 1.
        else:
            features = counts
        finish(name, features, frame, y, split,
               {'vocabulary_size': len(vectorizer.vocabulary_), 'min_df': 1, 'ngrams': [1,1],
                'preprocess_seconds': common_preprocess_seconds+time.perf_counter()-start, 'representation': name,
                'vocabulary_fit': 'NYT train only', 'feature_dimension': features.shape[1]}, start-common_preprocess_seconds,
               {'vectorizer': vectorizer, 'binary': name == 'bow_binary'})

def mean_embeddings(documents, vectors):
    features = np.zeros((len(documents), 100), dtype=np.float32)
    covered, total, zero = 0, 0, 0
    for i, tokens in enumerate(documents):
        valid = [vectors[word] for word in tokens if word in vectors]
        covered += len(valid)
        total += len(tokens)
        if valid: features[i] = np.mean(valid, axis=0)
        else: zero += 1
    return features, {'covered_tokens': covered, 'total_tokens': total,
                      'token_coverage': covered/max(total,1), 'zero_vector_documents': zero}

def embeddings(which):
    frame, y, split = load_data()
    docs = [tokenize(text) for text in frame.text]
    for name in which:
        start = time.perf_counter()
        metadata = {'feature_dimension': 100, 'pooling': 'unweighted mean over known token occurrences'}
        if name == 'glove':
            path = ROOT/'resources'/'glove.6B.100d.txt'
            if not path.exists(): raise FileNotFoundError('Run src/download_resources.py glove first')
            vectors = {}
            with path.open(encoding='utf-8') as source:
                for line in source:
                    word, _, values = line.partition(' ')
                    vector = np.fromstring(values, sep=' ', dtype=np.float32)
                    if vector.shape != (100,): raise ValueError('Invalid GloVe dimension')
                    vectors[word] = vector
            metadata.update({'source': 'Stanford glove.6B.100d.txt', 'loaded_vocabulary_size': len(vectors),
                             'vocabulary_filter': 'none; fixed public pretrained vectors'})
            joblib.dump(vectors, ROOT/'models'/'glove_vectors.joblib')
        else:
            from gensim.models import Word2Vec
            if name == 'w2v_ag':
                corpus = [tokenize(text) for text in pd.read_csv(ROOT/'HW-1'/'ag.csv').text]
                corpus_name = 'AG News only'
            else:
                corpus = [docs[ix] for ix in split['train']]
                corpus_name = 'NYT training split only'
            settings = dict(vector_size=100, window=5, min_count=2, sg=1, hs=0, negative=5,
                            sample=1e-3, epochs=10, workers=1, seed=SEED, hashfxn=stable_hash,
                            sorted_vocab=1)
            model = Word2Vec(sentences=corpus, **settings)
            vectors = model.wv
            model.save(str(ROOT/'models'/f'{name}.model'))
            metadata.update({'corpus': corpus_name, 'corpus_documents': len(corpus),
                             'gensim_version': version('gensim'),
                             'corpus_tokens': sum(map(len,corpus)), 'vocabulary_size': len(vectors),
                             'word2vec_settings': {k:v for k,v in settings.items() if k != 'hashfxn'}})
        features, coverage = mean_embeddings(docs, vectors)
        metadata['coverage'] = {key: mean_embeddings([docs[i] for i in ix], vectors)[1]
                                for key, ix in split.items()}
        scaler = StandardScaler().fit(features[split['train']])
        features = scaler.transform(features)
        metadata['standardization'] = 'mean and standard deviation fitted on NYT train only'
        metadata['embedding_seconds'] = time.perf_counter()-start
        finish(name, features, frame, y, split, metadata, start, {'scaler': scaler})

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['bow','glove','w2v_ag','w2v_nyt','all'], default='all')
    args = parser.parse_args()
    (ROOT/'models').mkdir(exist_ok=True)
    if args.task in ['bow','all']: bow()
    names = ['glove','w2v_ag','w2v_nyt'] if args.task == 'all' else [args.task]
    if args.task != 'bow': embeddings(names)
