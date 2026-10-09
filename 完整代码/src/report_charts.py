"""Plot numerical experiment results without article text or publisher imagery."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, LABELS
from analyze_results import NAMES

DISPLAY=['Binary BoW','Word Frequency','GloVe','W2V External','W2V Target','BERT']

def main():
    out=ROOT/'report'/'assets'; out.mkdir(parents=True,exist_ok=True)
    results={n:json.loads((ROOT/'results'/f'{n}.json').read_text(encoding='utf-8')) for n in NAMES}
    audit=json.loads((ROOT/'results'/'data_audit.json').read_text(encoding='utf-8'))
    stats=json.loads((ROOT/'results'/'statistical_analysis.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'semibold','axes.labelcolor':'#263746','text.color':'#263746'})
    palette=['#456D96','#C58946','#608E84']
    def save(fig,name): fig.savefig(out/name,dpi=220,bbox_inches='tight',facecolor='white'); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,3.2),layout='constrained')
    x=np.arange(3)
    for i,key in enumerate(['train','validation','test']):
        values=[audit['split'][key]['class_counts'][label]/audit['split'][key]['n']*100 for label in LABELS]
        ax.bar(x+(i-1)*.24,values,width=.23,color=palette[i],label=key.title())
    ax.set_xticks(x,[l.title() for l in LABELS]); ax.set_ylabel('Class proportion (%)'); ax.set_ylim(0,85)
    ax.legend(ncol=3,frameon=False); save(fig,'class_distribution.png')
    fig,ax=plt.subplots(figsize=(8.5,4),layout='constrained'); y=np.arange(6)
    for offset,key,color in [(-.17,'accuracy',palette[0]),(.17,'macro_f1',palette[1])]:
        values=[results[n]['test'][key]*100 for n in NAMES]
        ax.barh(y+offset,values,height=.31,color=color,label='Accuracy' if key=='accuracy' else 'Macro-F1')
        for i,value in enumerate(values): ax.text(value+.35,i+offset,f'{value:.2f}',va='center',fontsize=11)
    ax.set_yticks(y,DISPLAY); ax.invert_yaxis(); ax.set_xlim(0,105); ax.set_xlabel('Test score (%)')
    ax.legend(ncol=2,loc='lower right',bbox_to_anchor=(1,1.01),frameon=False); save(fig,'model_scores.png')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.1),layout='constrained')
    for ax,name,title in zip(axes,['bow_binary','bow_frequency'],DISPLAY[:2]):
        for weight,color in [(None,palette[0]),('balanced',palette[1])]:
            trials=[t for t in results[name]['trials'] if t['class_weight']==weight]
            ax.plot([t['C'] for t in trials],[t['validation_macro_f1']*100 for t in trials],'o-',color=color,label=str(weight))
        ax.set_xscale('log'); ax.set_xlabel('C'); ax.set_ylabel('Validation Macro-F1 (%)'); ax.set_title(title)
        ax.legend(title='Class weight',frameon=False,fontsize=10)
    save(fig,'validation_search.png')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.1),layout='constrained')
    history=results['bert']['history']; epochs=[r['epoch'] for r in history]
    axes[0].plot(epochs,[r['train_loss'] for r in history],'o-',color=palette[0],label='Train')
    axes[0].plot(epochs,[r['validation_loss'] for r in history],'s-',color=palette[1],label='Validation')
    axes[0].set_ylabel('Cross-entropy loss'); axes[0].legend(frameon=False)
    axes[1].plot(epochs,[r['validation_accuracy']*100 for r in history],'o-',color=palette[0],label='Accuracy')
    axes[1].plot(epochs,[r['validation_macro_f1']*100 for r in history],'s-',color=palette[1],label='Macro-F1')
    axes[1].set_ylabel('Validation score (%)'); axes[1].legend(frameon=False)
    for ax in axes: ax.set_xticks(epochs); ax.set_xlabel('Epoch')
    save(fig,'learning_curve.png')
    fig,ax=plt.subplots(figsize=(8.5,3.3),layout='constrained')
    matrix=np.array([[results[n]['test']['classification_report'][label]['f1-score']*100 for label in LABELS] for n in NAMES])
    ax.imshow(matrix,vmin=90,vmax=100,cmap='Blues',aspect='auto')
    for i in range(6):
        for j in range(3): ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]>96 else '#263746')
    ax.set_xticks(range(3),[l.title() for l in LABELS]); ax.set_yticks(range(6),DISPLAY)
    ax.set_title('Per-class F1 (%)'); save(fig,'class_f1.png')
    fig,axes=plt.subplots(2,3,figsize=(9,5.5),layout='constrained')
    for ax,name,title in zip(axes.flat,NAMES,DISPLAY):
        matrix=np.array(results[name]['test']['confusion_matrix']); norm=matrix/matrix.sum(axis=1,keepdims=True)
        ax.imshow(norm,cmap='Blues',vmin=0,vmax=1)
        for i in range(3):
            for j in range(3): ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if norm[i,j]>.5 else '#263746',fontsize=13)
        ax.set_title(title,fontsize=12); ax.set_xticks(range(3),['Bus.','Pol.','Sport'],fontsize=10); ax.set_yticks(range(3),['Bus.','Pol.','Sport'],fontsize=10)
        ax.set_xlabel('Predicted',fontsize=11); ax.set_ylabel('True',fontsize=11)
    save(fig,'all_confusions.png')
    fig,ax=plt.subplots(figsize=(8.5,3.1),layout='constrained')
    names=NAMES[2:5]; values=[results[n]['metadata']['coverage']['test']['token_coverage']*100 for n in names]
    ax.bar(range(3),values,color=palette); ax.set_xticks(range(3),['GloVe','External corpus','Target corpus'])
    ax.set_ylim(0,105); ax.set_ylabel('Known token occurrences (%)')
    for i,value in enumerate(values): ax.text(i,value+1,f'{value:.2f}%',ha='center')
    save(fig,'embedding_coverage.png')
    fig,ax=plt.subplots(figsize=(8.5,3.3),layout='constrained'); y=np.arange(6)
    values=np.array([results[n]['test']['macro_f1']*100 for n in NAMES])
    intervals=np.array([stats['confidence_intervals'][n]['macro_f1_95ci'] for n in NAMES])*100
    ax.errorbar(values,y,xerr=np.vstack([values-intervals[:,0],intervals[:,1]-values]),fmt='o',color=palette[0],capsize=4)
    ax.set_yticks(y,DISPLAY); ax.invert_yaxis(); ax.set_xlim(90,100); ax.set_xlabel('Macro-F1 (%) with 95% bootstrap intervals')
    save(fig,'score_intervals.png')
    fig,axes=plt.subplots(1,2,figsize=(8.5,3),layout='constrained')
    corpus=[results[n]['metadata'] for n in ['w2v_ag','w2v_nyt']]
    for ax,key,label,divisor in [(axes[0],'corpus_documents','Documents (thousands)',1000),(axes[1],'corpus_tokens','Token occurrences (millions)',1e6)]:
        vals=[r[key]/divisor for r in corpus]; ax.bar([0,1],vals,color=palette[:2]); ax.set_xticks([0,1],['External','Target']); ax.set_ylabel(label)
        for i,value in enumerate(vals): ax.text(i,value*1.015,f'{value:.2f}',ha='center')
        ax.set_ylim(0,max(vals)*1.16)
    save(fig,'corpus_sizes.png')
    print('Created 9 numerical charts',flush=True)

if __name__=='__main__': main()
