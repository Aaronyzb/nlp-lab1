"""Predict a single document using any saved HW1 model."""
import argparse
import joblib
import numpy as np
from common import ROOT, LABELS, tokenize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['bow_binary','bow_frequency','glove','w2v_ag','w2v_nyt','bert'])
    parser.add_argument('--text', required=True)
    args = parser.parse_args()
    if args.model == 'bert':
        import torch
        from transformers import BertTokenizerFast, BertForSequenceClassification
        folder = ROOT/'models'/'bert_best'
        tokenizer = BertTokenizerFast.from_pretrained(folder, local_files_only=True)
        model = BertForSequenceClassification.from_pretrained(folder, local_files_only=True)
        model.eval()
        with torch.no_grad():
            logits = model(**tokenizer(args.text, max_length=64, truncation=True, return_tensors='pt')).logits
            probabilities = torch.softmax(logits, dim=-1).numpy()[0]
    else:
        saved = joblib.load(ROOT/'models'/f'{args.model}_lr.joblib')
        transform = saved['transform']
        if args.model.startswith('bow_'):
            features = transform['vectorizer'].transform([args.text])
            if transform['binary']: features.data[:] = 1
        else:
            if args.model == 'glove': vectors = joblib.load(ROOT/'models'/'glove_vectors.joblib')
            else:
                from gensim.models import Word2Vec
                vectors = Word2Vec.load(str(ROOT/'models'/f'{args.model}.model')).wv
            from train_classical import mean_embeddings
            features = transform['scaler'].transform(mean_embeddings([tokenize(args.text)], vectors)[0])
        probabilities = saved['classifier'].predict_proba(features)[0]
    print(LABELS[int(np.argmax(probabilities))])
    print(dict(zip(LABELS, map(float,probabilities))))

if __name__ == '__main__': main()
