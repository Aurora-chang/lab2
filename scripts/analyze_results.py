"""Rebuild portable experiment tables and figures from results/runs snapshots.

Optional --checkpoint recomputes confusion matrices using the selected checkpoint.
No training is performed. Published summaries never depend on local absolute paths.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/lab2-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
SPLITS=('train','validation','test')


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def load_runs():
    runs=[]
    for folder in sorted((RESULTS/'runs').iterdir()):
        if not folder.is_dir(): continue
        config=json.loads((folder/'config.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        with (folder/'metrics.csv').open() as stream:
            metrics=[{key:float(value) for key,value in row.items()} for row in csv.DictReader(stream)]
        if len(metrics)!=150 or [int(m['epoch']) for m in metrics]!=list(range(1,151)):
            raise ValueError(f'Expected 150 complete epochs: {folder.name}')
        if not all(np.isfinite(list(m.values())).all() for m in metrics):
            raise ValueError('Non-finite metrics')
        chosen=max(metrics,key=lambda m:m['validation_accuracy'])
        peak=max(metrics,key=lambda m:m['test_accuracy'])
        if chosen['epoch']!=summary['selected_epoch'] or peak['test_accuracy']!=summary['peak_test_accuracy'] or peak['epoch']!=summary['peak_test_epoch']:
            raise ValueError(f'Incorrect checkpoint/peak summary: {folder.name}')
        for split in ('validation','test'):
            for metric in ('accuracy','loss'):
                if not np.isclose(chosen[f'{split}_{metric}'],summary['selected_checkpoint_scores'][split][metric],atol=1e-5,rtol=1e-4):
                    raise ValueError(f'Selected metric mismatch: {folder.name}')
        if config['experiment_name']!=folder.name or config['evaluation']['split_seed']!=42:
            raise ValueError('Inconsistent experiment identity or split')
        runs.append({'folder':folder,'config':config,'summary':summary,'metrics':metrics})
    if len(runs)!=17: raise ValueError('Expected the 17 reviewed full experiments')
    return runs


def tables(runs):
    rows=[]; losses=[]
    for run in runs:
        c,s=run['config'],run['summary']
        row={'experiment':c['experiment_name'],'run_dir':str(run['folder'].relative_to(ROOT)),
             'seed':c['seed'],'split_seed':42,'epochs':150,**c['model'],
             'learning_rate':c['training']['learning_rate'],'selected_epoch':s['selected_epoch'],
             'validation_accuracy':s['selected_checkpoint_scores']['validation']['accuracy'],
             'selected_test_accuracy':s['selected_checkpoint_scores']['test']['accuracy'],
             'peak_monitored_test_accuracy':s['peak_test_accuracy'],'peak_test_epoch':s['peak_test_epoch'],
             'parameters':s['parameters'],'duration_seconds':s['duration_seconds']}
        loss={'experiment':c['experiment_name'],'seed':c['seed'],'selected_epoch':s['selected_epoch']}
        for split in SPLITS:
            row[f'{split}_loss']=s['selected_checkpoint_scores'][split]['loss']
            loss[f'selected_checkpoint_{split}_loss']=s['selected_checkpoint_scores'][split]['loss']
            loss[f'final_epoch_logged_{split}_loss']=run['metrics'][-1][f'{split}_loss']
        rows.append(row); losses.append(loss)
    rows.sort(key=lambda r:-r['validation_accuracy'])
    write_csv(RESULTS/'experiments.csv',rows); write_csv(RESULTS/'loss_comparison.csv',losses)
    stats=[]
    for group,prefix in [('baseline','elu_baseline_seed'),('leaky_relu','leaky_relu_seed')]:
        chosen=[r for r in rows if r['experiment'].startswith(prefix)]
        if sorted(r['seed'] for r in chosen)!=[42,43,44]: raise ValueError('Missing/duplicate seed')
        stat={'configuration':group,'n':3,'model_seeds':'42,43,44','split_seed':42}
        for metric in ['validation_accuracy','selected_test_accuracy']:
            values=[r[metric] for r in chosen]
            stat[metric+'_mean']=statistics.mean(values)
            stat[metric+'_sample_std']=statistics.stdev(values)
        stats.append(stat)
    write_csv(RESULTS/'seed_summary.csv',stats)
    return rows,stats


def curves(run,path):
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    epochs=[m['epoch'] for m in run['metrics']]
    for ax,metric in zip(axes,['loss','accuracy']):
        for split in SPLITS: ax.plot(epochs,[m[f'{split}_{metric}'] for m in run['metrics']],label=split)
        ax.axvline(run['summary']['selected_epoch'],color='black',linestyle=':',alpha=.6,label='Selected epoch')
        ax.set(xlabel='Epoch',ylabel='Cross-entropy loss' if metric=='loss' else 'Accuracy (%)')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle(run['config']['experiment_name']); fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def figures(runs,rows,stats):
    destination=RESULTS/'figures'; destination.mkdir(exist_ok=True)
    by_name={r['config']['experiment_name']:r for r in runs}
    for name,filename in [('elu_baseline_seed42','baseline_curves.png'),('leaky_relu_seed42','selected_curves.png'),('leaky_relu_seed43','selected_seed43_curves.png'),('leaky_relu_seed44','selected_seed44_curves.png')]:
        curves(by_name[name],destination/filename)
    fig,axes=plt.subplots(6,3,figsize=(16,22),sharex=True)
    for ax,run in zip(axes.flat,runs):
        for split in SPLITS: ax.plot(range(1,151),[m[f'{split}_loss'] for m in run['metrics']],label=split,linewidth=1)
        ax.axvline(run['summary']['selected_epoch'],color='black',linestyle=':',alpha=.5)
        ax.set(title=run['config']['experiment_name'].replace('_seed','\nseed ').replace('_',' '),xlabel='Epoch',ylabel='Cross-entropy loss'); ax.grid(alpha=.2)
    for ax in list(axes.flat)[len(runs):]: ax.set_visible(False)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.975))
    fig.suptitle('All 17 experiments: train / validation / test loss\nDotted line: validation-selected epoch. Train metrics use training mode.',y=.997)
    fig.tight_layout(rect=(0,0,1,.955)); fig.savefig(destination/'all_loss_curves.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,metric,title in zip(axes,['validation_accuracy','selected_test_accuracy'],['Validation accuracy','Selected-checkpoint test accuracy']):
        for seed in [42,43,44]:
            values=[next(r[metric] for r in rows if r['experiment']==f'{family}_seed{seed}') for family in ['elu_baseline','leaky_relu']]
            ax.plot([0,1],values,'o-',alpha=.65,label=f'Seed {seed}')
        ax.errorbar([.08,1.08],[s[metric+'_mean'] for s in stats],yerr=[s[metric+'_sample_std'] for s in stats],fmt='ks',capsize=5,label='Mean ± sample SD')
        ax.set(xticks=[0,1],xticklabels=['ELU baseline','LeakyReLU'],ylabel='Accuracy (%)',title=title,xlim=(-.25,1.35),ylim=(70,92)); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Model seeds 42–44; fixed split seed 42'); fig.tight_layout(); fig.savefig(destination/'seed_comparison.png',dpi=160); plt.close(fig)
    for filename,names,title in [
        ('combination_comparison.png',['leaky_relu_seed42','leaky_dropout_05_seed42','leaky_lr_0001_seed42','leaky_lr_0005_seed42'],'LeakyReLU combinations, seed 42'),
        ('initial_comparison.png',[r['experiment'] for r in rows if r['seed']==42 and not r['experiment'].startswith('leaky_')]+['leaky_relu_seed42'],'One-factor comparisons, seed 42')]:
        chosen=sorted([r for r in rows if r['experiment'] in names],key=lambda r:-r['validation_accuracy'])
        fig,ax=plt.subplots(figsize=(10,max(4,len(chosen)*.5)))
        y=np.arange(len(chosen)); ax.barh(y-.18,[r['validation_accuracy'] for r in chosen],height=.36,label='Validation'); ax.barh(y+.18,[r['selected_test_accuracy'] for r in chosen],height=.36,label='Selected-checkpoint test')
        ax.set_yticks(y,[r['experiment'].removesuffix('_seed42').replace('_',' ') for r in chosen]); ax.invert_yaxis(); ax.set(xlabel='Accuracy (%)',xlim=(0,100),title=title); ax.legend(); fig.tight_layout(); fig.savefig(destination/filename,dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name in ['elu_baseline_seed42','leaky_relu_seed42','relu_seed42','lr_0001_seed42']:
        for ax,metric in zip(axes,['validation_accuracy','validation_loss']): ax.plot(range(1,151),[m[metric] for m in by_name[name]['metrics']],label=name,alpha=.8,linewidth=1)
    for ax,label in zip(axes,['Validation accuracy (%)','Validation loss']): ax.set(xlabel='Epoch',ylabel=label); ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(destination/'validation_comparison.png',dpi=160); plt.close(fig)


def infer_confusion(checkpoint_path,data_dir,run):
    import torch
    sys.path.insert(0,str(ROOT/'lab2_EEG_classification'))
    from dataloader import prepare_data
    from main import source_hashes
    from models.EEGNet import EEGNet
    torch.set_num_threads(2)
    expected=json.loads((RESULTS/'confusion_matrices.json').read_text())
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()!=expected['checkpoint_sha256']: raise ValueError('Checkpoint hash differs')
    environment=json.loads((run['folder']/'environment.json').read_text())
    if source_hashes()!=environment['source_sha256']: raise ValueError('Training source differs')
    if {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in data_dir.glob('*.npz')}!=environment['dataset_sha256']: raise ValueError('Dataset differs')
    c=run['config']; e=c['evaluation']; splits,prep=prepare_data(data_dir,e['validation_fraction'],e['split_seed'],e['stratify'])
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    if checkpoint['preprocessing']!=prep or checkpoint['epoch']!=run['summary']['selected_epoch']: raise ValueError('Checkpoint preprocessing/epoch differs')
    model=EEGNet(**{k:v for k,v in c['model'].items() if k!='name'}); model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    for split,(x,y) in splits.items():
        matrix=np.zeros((2,2),dtype=int); total=0.
        with torch.inference_mode():
            for start in range(0,len(y),c['training']['batch_size']):
                end=start+c['training']['batch_size']; targets=torch.from_numpy(y[start:end]); logits=model(torch.from_numpy(x[start:end]))
                np.add.at(matrix,(y[start:end],logits.argmax(1).numpy()),1)
                total+=torch.nn.functional.cross_entropy(logits,targets,reduction='sum').item()
        scores=run['summary']['selected_checkpoint_scores'][split]
        if not np.isclose(100*np.trace(matrix)/matrix.sum(),scores['accuracy'],atol=1e-6,rtol=0) or not np.isclose(total/len(y),scores['loss'],atol=1e-5,rtol=1e-4): raise ValueError('Inference does not reproduce recorded metrics')
        if matrix.tolist()!=expected['splits'][split]['counts']: raise ValueError('Confusion counts differ')
    print('Checkpoint inference reproduced confusion counts, loss, and accuracy on every split.')


def confusion_figure(run):
    saved=json.loads((RESULTS/'confusion_matrices.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,split in zip(axes,SPLITS):
        matrix=np.asarray(saved['splits'][split]['counts']); normalized=matrix/matrix.sum(axis=1,keepdims=True)*100
        if not np.isclose(np.trace(matrix)/matrix.sum()*100,run['summary']['selected_checkpoint_scores'][split]['accuracy']): raise ValueError('Confusion accuracy differs')
        ax.imshow(normalized,cmap='Blues',vmin=0,vmax=100)
        for i in range(2):
            for j in range(2): ax.text(j,i,f'{matrix[i,j]}\n({normalized[i,j]:.1f}%)',ha='center',va='center',color='white' if normalized[i,j]>50 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted label',ylabel='True label',title=f'{split.title()} (n={matrix.sum()})\nAccuracy {100*np.trace(matrix)/matrix.sum():.2f}%')
    fig.suptitle('Selected LeakyReLU: seed 42, validation-selected epoch 131\nCounts and row-normalized percentages; labels 0/1 map to original labels 1/2',fontsize=11)
    fig.tight_layout(); fig.savefig(RESULTS/'figures/confusion_matrices.png',dpi=180); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,help='Optional original seed-42 best.pt for independent inference')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'lab2_EEG_classification/data')
    args=parser.parse_args()
    runs=load_runs(); rows,stats=tables(runs); figures(runs,rows,stats)
    selected=next(r for r in runs if r['config']['experiment_name']=='leaky_relu_seed42')
    if args.checkpoint: infer_confusion(args.checkpoint,args.data_dir,selected)
    confusion_figure(selected)
    print('Validated 17 snapshots and rebuilt tables/figures. No training performed.')


if __name__=='__main__': main()
