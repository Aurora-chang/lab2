"""Plot and independently review DeepConvNet bonus artifacts without training."""
import argparse
import csv
import json
import hashlib
import shutil
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/lab2-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lab2_EEG_classification'))
sys.path.insert(0,str(ROOT/'scripts'))
from main import plot_curves,write_json
from run_bonus import CANDIDATE_PATHS,identity,select_candidate,summarize_selected
from run_experiments import verify_saved
from bonus_training import load_config


def run_folder(output,name):
    snapshot=output/'runs'/name
    if (snapshot/'summary.json').exists() and json.loads((snapshot/'config.json').read_text())['experiment_name']==name:
        return snapshot
    found=[p.parent for p in output.glob(name+'_*/config.json')
           if json.loads(p.read_text())['experiment_name']==name and (p.parent/'summary.json').exists()]
    if len(found)!=1: raise ValueError(f'Expected one completed run for {name}')
    return found[0]


def create_figures(output):
    rows=json.loads((output/'comparison.json').read_text())
    selection=json.loads((output/'selection.json').read_text())
    figures=output/'figures'; figures.mkdir(exist_ok=True)
    family=selection['selected_experiment'].removesuffix('_seed42')
    for seed in [42,43,44]:
        folder=run_folder(output,family+f'_seed{seed}')
        with (folder/'metrics.csv').open() as stream:
            metrics=[{k:float(v) if v else None for k,v in row.items()} for row in csv.DictReader(stream)]
        plot_curves(metrics,figures/f'deepconvnet_seed{seed}_curves.png')
    folder=run_folder(output,selection['selected_experiment'])
    matrices=json.loads((folder/'confusion_matrices.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,split in zip(axes,['train','validation','test']):
        counts=np.asarray(matrices['splits'][split]); normalized=counts/counts.sum(1,keepdims=True)*100
        ax.imshow(normalized,cmap='Blues',vmin=0,vmax=100)
        for i in range(2):
            for j in range(2): ax.text(j,i,f'{counts[i,j]}\n({normalized[i,j]:.1f}%)',ha='center',va='center',color='white' if normalized[i,j]>50 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted label',ylabel='True label',title=f'{split.title()} (n={counts.sum()})\nAccuracy {100*np.trace(counts)/counts.sum():.2f}%')
    fig.suptitle(f'DeepConvNet: seed 42, validation-selected epoch {matrices["selected_epoch"]}\nCounts and row-normalized percentages; model labels 0/1 = original labels 1/2',fontsize=10)
    fig.tight_layout(); fig.savefig(figures/'deepconvnet_confusion_matrices.png',dpi=160); plt.close(fig)
    with (output/'seed_summary.csv').open() as stream: stats=list(csv.DictReader(stream))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric,title in zip(axes,['validation_accuracy','selected_test_accuracy'],['Validation accuracy','Selected-checkpoint test accuracy']):
        ax.bar(range(len(stats)),[float(s[metric+'_mean']) for s in stats],yerr=[float(s[metric+'_sample_std']) for s in stats],capsize=5)
        ax.set(xticks=range(len(stats)),xticklabels=[s['model']+'\n'+s['configuration'] for s in stats],ylabel='Accuracy (%)',title=title,ylim=(0,100)); ax.tick_params(axis='x',labelsize=7)
    fig.suptitle('Model seeds 42, 43, 44: mean ± sample SD; fixed split seed 42')
    fig.tight_layout(); fig.savefig(figures/'eegnet_deepconvnet_comparison.png',dpi=160); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
    sources=[(r['experiment'],run_folder(output,r['experiment'])) for r in rows]
    sources.append(('EEGNet selected, seed 42',ROOT/'results/runs/leaky_relu_seed42'))
    for ax,(name,folder) in zip(axes.flat,sources):
        with (folder/'metrics.csv').open() as stream: metrics=list(csv.DictReader(stream))
        for split in ['train','validation','test']: ax.plot([int(r['epoch']) for r in metrics],[float(r[f'{split}_loss']) for r in metrics],label=split,linewidth=1)
        ax.set(title=name.replace('_seed','\nseed ').replace('_',' '),xlabel='Epoch',ylabel='Cross-entropy loss'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Loss comparison; training curve uses training mode')
    fig.tight_layout(); fig.savefig(figures/'loss_comparison.png',dpi=160); plt.close(fig)


def review(output,device='cpu'):
    selection=json.loads((output/'selection.json').read_text())
    complete=json.loads((output/'complete.json').read_text())
    rows=json.loads((output/'comparison.json').read_text())
    if len(rows)!=5 or complete['new_full_runs']!=5: raise ValueError('Expected five full bonus runs')
    configs=[load_config(p,output) for p in CANDIDATE_PATHS]
    expected={c['experiment_name']:c for c in configs}
    winner=select_candidate(rows,list(expected))
    if winner['experiment']!=selection['selected_experiment'] or winner['experiment']!=complete['selected_experiment']: raise ValueError('Selection was not validation-based')
    if {r['experiment']:r['validation_accuracy'] for r in rows if r['seed']==42}!=selection['candidate_validation_accuracies']:
        raise ValueError('Stored candidate scores differ')
    selected=expected[winner['experiment']]
    if identity(selection['selected_config'])!=identity(selected): raise ValueError('Selected settings differ')
    for seed in [43,44]:
        c=json.loads(json.dumps(selected)); c['seed']=seed; c['experiment_name']=winner['experiment'].removesuffix('_seed42')+f'_seed{seed}'
        expected[c['experiment_name']]=c
    if {r['experiment'] for r in rows}!=set(expected): raise ValueError('Unexpected or missing configurations')
    checked=[]
    for row in rows:
        folder=run_folder(output,row['experiment'])
        config=json.loads((folder/'config.json').read_text())
        if identity(config)!=identity(expected[row['experiment']]): raise ValueError('Config differs from protocol')
        gpu=json.loads((folder/'verification.json').read_text())
        if gpu['status']!='passed' or not gpu['verification_device'].startswith('cuda'): raise ValueError('Missing original GPU verification')
        checked.append(verify_saved(folder,ROOT/'lab2_EEG_classification/data',device))
        for key,value in {'seed':config['seed'],'split_seed':config['evaluation']['split_seed'],
                          'epochs':config['training']['epochs'],'activation':config['model']['activation'],
                          'learning_rate':config['training']['learning_rate'],'dropout':config['model']['dropout']}.items():
            if row[key]!=value: raise ValueError(f'Comparison configuration mismatch: {key}')
        summary=json.loads((folder/'summary.json').read_text())
        for key,expected_value in [('selected_epoch',summary['selected_epoch']),('validation_accuracy',summary['selected_checkpoint_scores']['validation']['accuracy']),('selected_test_accuracy',summary['selected_checkpoint_scores']['test']['accuracy']),('peak_monitored_test_accuracy',summary['peak_test_accuracy']),('parameters',summary['parameters'])]:
            if row[key]!=expected_value: raise ValueError(f'Comparison row mismatch: {key}')
        print('Verified:',row['experiment'],flush=True)
    with (output/'comparison.csv').open() as stream: csv_rows=list(csv.DictReader(stream))
    if len(csv_rows)!=len(rows) or any(any(saved[key]!=str(value) for key,value in row.items()) for saved,row in zip(csv_rows,rows)):
        raise ValueError('Comparison CSV differs from JSON')
    selected_stats=summarize_selected(rows,winner['experiment'])
    saved=json.loads((output/'seed_summary.json').read_text())
    if len(saved)!=3 or saved[-1]!=selected_stats: raise ValueError('DeepConvNet seed statistics differ')
    with (output/'seed_summary.csv').open() as stream: csv_stats=list(csv.DictReader(stream))
    if len(csv_stats)!=len(saved) or any(any(row[key]!=str(value) for key,value in reference.items()) for row,reference in zip(csv_stats,saved)):
        raise ValueError('Seed-summary CSV differs from JSON')
    with (ROOT/'results/seed_summary.csv').open() as stream: historical=list(csv.DictReader(stream))
    for row,reference in zip(saved[:-1],historical):
        for key,value in reference.items():
            if str(row[key])!=value: raise ValueError('EEGNet reference differs')
    create_figures(output)
    result={'status':'passed','new_full_runs':5,'selected_experiment':winner['experiment'],
            'selected_statistics':selected_stats,'checks':checked,
            'note':'GPU batch verified locally; interpretation/comparison can now be finalized.'}
    write_json(output/'local_review.json',result)
    print('All bonus runs and seed statistics verified.')


def export_public(output,destination):
    """Export reviewed, weight-free snapshots with portable paths for GitHub."""
    audit=json.loads((output/'local_review.json').read_text())
    if audit['status']!='passed' or audit['new_full_runs']!=5:
        raise ValueError('Complete independent review required before public export')
    rows=json.loads((output/'comparison.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    portable=[]
    for row in rows:
        source=run_folder(output,row['experiment'])
        target=destination/'runs'/row['experiment']; target.mkdir(parents=True,exist_ok=True)
        config=json.loads((source/'config.json').read_text())
        config.update(data_dir='lab2_EEG_classification/data',output_dir='outputs/bonus')
        write_json(target/'config.json',config)
        for name in ['metrics.csv','summary.json','environment.json','confusion_matrices.json']:
            shutil.copy2(source/name,target/name)
        gpu=json.loads((source/'verification.json').read_text())
        cpu=json.loads((source/'local_review.json').read_text())
        for value in [gpu,cpu]: value.pop('run_dir',None)
        write_json(target/'verification.json',{
            'checkpoint_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ['best.pt','last.pt']},
            'gpu_verification':gpu,'independent_cpu_verification':cpu})
        item=dict(row); item['run_dir']=str(target.relative_to(ROOT)); portable.append(item)
    write_json(destination/'comparison.json',portable)
    with (destination/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(portable[0])); writer.writeheader(); writer.writerows(portable)
    for name in ['selection.json','seed_summary.json','seed_summary.csv']:
        shutil.copy2(output/name,destination/name)
    write_json(destination/'verification.json',{
        'status':'passed','reviewed_full_runs':5,'selected_experiment':audit['selected_experiment'],
        'source':'Independent CPU review of original GPU runs',
        'checks':['source_and_data_hashes','planned_configs','fixed_split','checkpoint_predictions',
                  'all_split_confusion_counts','max_norm_constraints','validation_selection',
                  'csv_json_consistency','sample_standard_deviation'],
        'selected_statistics':audit['selected_statistics']})
    create_figures(destination)
    print('Public bonus snapshots and figures exported to',destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--export',type=Path,help='Export verified raw results to a public directory')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    output=args.output.resolve()
    if args.verify:
        review(output,args.device)
    elif not args.export:
        create_figures(output)
    if args.export:
        export_public(output,args.export.resolve())
