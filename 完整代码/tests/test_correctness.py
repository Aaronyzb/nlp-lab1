"""Check token counts, embedding averages, data splits and saved metrics."""
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from common import ROOT, LABELS, tokenize, load_data, metrics
from train_classical import mean_embeddings

class CorrectnessTests(unittest.TestCase):
    def test_tokenizer(self):
        self.assertEqual(tokenize("Apple apple, DON'T 12.5!"), ['apple','apple',"don't",'12.5'])

    def test_bow_counts_binary_and_unseen(self):
        vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
        counts = vectorizer.fit_transform(['apple apple banana','banana'])
        self.assertEqual(counts.toarray().tolist(), [[2,1],[0,1]])
        binary = counts.copy(); binary.data[:] = 1
        self.assertEqual(binary.toarray().tolist(), [[1,1],[0,1]])
        self.assertEqual(vectorizer.transform(['testonly apple']).toarray().tolist(), [[1,0]])
        self.assertNotIn('testonly',vectorizer.vocabulary_)

    def test_average_repetitions_oov_and_empty(self):
        vectors = {'apple': np.ones(100), 'banana': np.ones(100)*4}
        values, coverage = mean_embeddings([['apple','apple','banana','unknown'],['unknown'],[]], vectors)
        np.testing.assert_allclose(values[0],2.)
        np.testing.assert_array_equal(values[1:],0.)
        self.assertEqual(coverage['zero_vector_documents'],2)
        self.assertEqual(coverage['covered_tokens'],3)

    def test_split_is_disjoint_complete_reproducible(self):
        frame,y,split = load_data()
        self.assertEqual([len(split[k]) for k in ['train','validation','test']], [9215,1152,1152])
        np.testing.assert_array_equal(np.sort(np.concatenate(list(split.values()))),np.arange(len(frame)))
        _,_,again = load_data()
        for key in split: np.testing.assert_array_equal(split[key],again[key])

    def test_macro_f1_includes_all_classes(self):
        result = metrics([0,1,2,2],[0,2,2,2])
        self.assertAlmostEqual(result['accuracy'],.75)
        self.assertAlmostEqual(result['macro_f1'],.6)

    def test_saved_predictions_reproduce_metrics_and_split(self):
        _, y, split = load_data()
        files = list((ROOT/'results').glob('*_predictions.csv'))
        expected={'bow_binary','bow_frequency','glove','w2v_ag','w2v_nyt','bert'}
        self.assertEqual({p.stem.removesuffix('_predictions') for p in files},expected,
                         'Complete all six experiments before the final audit')
        for path in files:
            table = pd.read_csv(path)
            np.testing.assert_array_equal(table.row_id, split['test'])
            true = table.true_label.map(dict(zip(LABELS,range(3)))).to_numpy()
            pred = table.predicted_label.map(dict(zip(LABELS,range(3)))).to_numpy()
            np.testing.assert_array_equal(true,y[split['test']])
            result = json.loads(path.with_name(path.name.replace('_predictions.csv','.json')).read_text(encoding='utf-8'))
            recomputed = metrics(true,pred)
            self.assertEqual(recomputed,result['test'])
            self.assertEqual(np.sum(result['test']['confusion_matrix']),1152)

    def test_bert_and_embedding_assignment_constraints(self):
        result=json.loads((ROOT/'results'/'bert.json').read_text(encoding='utf-8'))
        self.assertEqual(result['metadata']['max_length'],64)
        self.assertEqual(result['metadata']['epochs'],3)
        self.assertEqual([epoch['epoch'] for epoch in result['history']],[1,2,3])
        self.assertTrue(all(epoch['samples_seen']==9215 for epoch in result['history']))
        self.assertGreater(result['metadata']['trainable_parameters'],100_000_000)
        best=max(result['history'],key=lambda epoch:(epoch['validation_macro_f1'],epoch['validation_accuracy']))
        self.assertEqual(result['selected_epoch'],best['epoch'])
        for name in ['glove','w2v_ag','w2v_nyt']:
            result=json.loads((ROOT/'results'/f'{name}.json').read_text(encoding='utf-8'))
            self.assertEqual(result['metadata']['feature_dimension'],100)
            self.assertEqual(result['metadata']['coverage']['test']['zero_vector_documents'],0)

if __name__ == '__main__': unittest.main()
