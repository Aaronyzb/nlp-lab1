"""Task 3: full BERT fine-tuning, exactly 3 epochs, maximum sequence length 64."""
import argparse
import contextlib
import json
import math
import os
import platform
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
from common import ROOT, SEED, LABELS, load_data, metrics, write_json, save_predictions

class EncodedDataset(Dataset):
    def __init__(self, encodings, labels, indices):
        self.encodings, self.labels, self.indices = encodings, labels, indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        index = self.indices[i]
        return {**{key: value[index] for key, value in self.encodings.items()},
                'labels': torch.tensor(self.labels[index], dtype=torch.long)}

def evaluate(model, loader, device, autocast):
    model.eval()
    targets, predictions, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key,value in batch.items()}
            with autocast(): output = model(**batch)
            targets.extend(batch['labels'].cpu().tolist())
            predictions.extend(output.logits.argmax(-1).cpu().tolist())
            losses.append((output.loss.item(), len(batch['labels'])))
    return metrics(targets, predictions), np.array(predictions), sum(l*n for l,n in losses)/sum(n for _,n in losses)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    args = parser.parse_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(8)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else
                          'cpu' if args.device == 'auto' else args.device)
    autocast = (lambda: torch.autocast('cuda', dtype=torch.float16)) if device.type == 'cuda' else contextlib.nullcontext
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    frame, y, split = load_data()
    local = ROOT/'resources'/'bert-base-uncased'
    if not (local/'model.safetensors').exists():
        raise FileNotFoundError('Run src/download_resources.py bert first')
    tokenizer = BertTokenizerFast.from_pretrained(str(local), local_files_only=True)
    encodings = tokenizer(frame.text.tolist(), truncation=True, padding='max_length',
                          max_length=64, return_tensors='pt')
    full_lengths = np.array([len(ids) for ids in tokenizer(frame.text.tolist(), truncation=False)['input_ids']])
    lengths = {key: {'fraction_truncated': float(np.mean(full_lengths[ix]>64)),
                     'mean_full_wordpiece_length': float(np.mean(full_lengths[ix]))}
               for key, ix in split.items()}
    loaders = {key: DataLoader(EncodedDataset(encodings, y, ix), batch_size=args.batch_size,
                               shuffle=key=='train', generator=torch.Generator().manual_seed(SEED),
                               num_workers=0, pin_memory=device.type=='cuda') for key,ix in split.items()}
    model = BertForSequenceClassification.from_pretrained(str(local), num_labels=3,
                id2label=dict(enumerate(LABELS)), label2id={label:i for i,label in enumerate(LABELS)},
                local_files_only=True, attn_implementation='eager').to(device)
    assert all(parameter.requires_grad for parameter in model.parameters()), 'Must fine-tune full BERT'
    # Standard AdamW: exempt biases and LayerNorm parameters from weight decay.
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer = torch.optim.AdamW([
        {'params': [p for n,p in model.named_parameters() if not any(s in n for s in no_decay)], 'weight_decay': .01},
        {'params': [p for n,p in model.named_parameters() if any(s in n for s in no_decay)], 'weight_decay': 0.}], lr=2e-5)
    steps = len(loaders['train'])*3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=math.ceil(.1*steps), num_training_steps=steps)
    directory = ROOT/'models'/'bert_best'
    history, best_rank, best_epoch = [], (-1.,-1.), None
    started = time.perf_counter()
    print(f'Device {device}, batch {args.batch_size}, steps per epoch {len(loaders["train"])}', flush=True)
    for epoch in range(1,4):
        model.train()
        total_loss, n_seen = 0., 0
        epoch_start = time.perf_counter()
        for step, batch in enumerate(loaders['train'],1):
            batch = {key:value.to(device) for key,value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast(): loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            old_scale = scaler.get_scale()
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale() >= old_scale: scheduler.step()
            total_loss += loss.item()*len(batch['labels']); n_seen += len(batch['labels'])
            if step%25 == 0 or step == len(loaders['train']):
                print(f'epoch {epoch}/3 step {step}/{len(loaders["train"])} loss {total_loss/n_seen:.5f} elapsed {time.perf_counter()-epoch_start:.1f}s', flush=True)
        val, _, val_loss = evaluate(model, loaders['validation'], device, autocast)
        record = {'epoch':epoch, 'train_loss':total_loss/n_seen, 'validation_loss':val_loss,
                  'validation_accuracy':val['accuracy'], 'validation_macro_f1':val['macro_f1'],
                  'epoch_seconds':time.perf_counter()-epoch_start, 'samples_seen':n_seen}
        history.append(record); print(record, flush=True)
        if (val['macro_f1'], val['accuracy']) > best_rank:
            best_rank, best_epoch = (val['macro_f1'],val['accuracy']), epoch
            model.save_pretrained(directory); tokenizer.save_pretrained(directory)
        write_json(ROOT/'results'/'bert_history.json',history)
    del optimizer, scheduler, model
    if device.type == 'cuda': torch.cuda.empty_cache()
    model = BertForSequenceClassification.from_pretrained(directory, local_files_only=True, attn_implementation='eager').to(device)
    test, prediction, test_loss = evaluate(model, loaders['test'], device, autocast)
    source = json.loads((ROOT/'resources'/'bert_source.json').read_text())
    result = {'name':'bert', 'test':test, 'test_loss':test_loss, 'history':history,
              'selected_epoch':best_epoch, 'selection':'highest validation macro-F1, then accuracy',
              'elapsed_seconds':time.perf_counter()-started,
              'metadata':{'source':source, 'max_length':64, 'epochs':3, 'batch_size':args.batch_size,
                          'learning_rate':2e-5, 'weight_decay':.01, 'warmup_ratio':.1, 'clip_norm':1.,
                          'seed':SEED, 'precision':'float16 AMP' if device.type=='cuda' else 'float32',
                          'device':str(device), 'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                          'torch_version':torch.__version__, 'platform':platform.platform(),
                          'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                          'truncation':lengths},
              'duplicate_sensitivity':save_predictions('bert',split['test'],y[split['test']],prediction,frame)}
    write_json(ROOT/'results'/'bert.json',result)
    print(f"FINISHED BERT accuracy={test['accuracy']:.6f} macro-F1={test['macro_f1']:.6f}",flush=True)

if __name__ == '__main__': main()
