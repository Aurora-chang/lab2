"""DeepConvNet search, validation-only selection, and seed repetitions.

Defaults: three seed-42 candidates, then the winner at seeds 43/44. Existing
EEGNet scores are compared without retraining. Run --smoke for local checks.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lab2_EEG_classification'))
sys.path.insert(0,str(ROOT/'scripts'))
import torch
from bonus_training import load_config,run
from main import write_json
from run_experiments import identity,verify_saved

CANDIDATE_PATHS=[ROOT/'configs/bonus/deepconvnet_elu.json',
                 ROOT/'configs/bonus/deepconvnet_elu_lr0005.json',
                 ROOT/'configs/bonus/deepconvnet_leaky_relu.json']


def select_candidate(rows,names):
    by_name={r['experiment']:r for r in rows}
    return max((by_name[name] for name in names),key=lambda row:row['validation_accuracy'])


def execute(config,device,threads):
    output=Path(config['output_dir']); output.mkdir(parents=True,exist_ok=True)
    matches=[p.parent for p in output.glob(config['experiment_name']+'_*/config.json')
             if identity(json.loads(p.read_text()))==identity(config) and (p.parent/'summary.json').exists()]
    if len(matches)>1: raise RuntimeError(f'Ambiguous completed runs: {matches}')
    folder=matches[0] if matches else run(config,torch.device(device),threads)
    verify_saved(folder,Path(config['data_dir']),device)
    s=json.loads((folder/'summary.json').read_text())
    return {'experiment':config['experiment_name'],'seed':config['seed'],'split_seed':config['evaluation']['split_seed'],
            'epochs':config['training']['epochs'],'activation':config['model']['activation'],
            'learning_rate':config['training']['learning_rate'],'dropout':config['model']['dropout'],
            'selected_epoch':s['selected_epoch'],'validation_accuracy':s['selected_checkpoint_scores']['validation']['accuracy'],
            'selected_test_accuracy':s['selected_checkpoint_scores']['test']['accuracy'],
            'peak_monitored_test_accuracy':s['peak_test_accuracy'],'peak_test_epoch':s['peak_test_epoch'],
            'parameters':s['parameters'],'duration_seconds':s['duration_seconds'],'run_dir':str(folder)}


def save_table(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def summarize_selected(rows,name):
    family=name.removesuffix('_seed42')
    selected=[r for r in rows if r['experiment'] in [family+f'_seed{seed}' for seed in [42,43,44]]]
    if sorted(r['seed'] for r in selected)!=[42,43,44]: raise ValueError('Missing selected seeds')
    result={'model':'DeepConvNet','configuration':family,'n':3,'model_seeds':'42,43,44','split_seed':42}
    for key in ['validation_accuracy','selected_test_accuracy']:
        values=[r[key] for r in selected]
        result[key+'_mean']=statistics.mean(values); result[key+'_sample_std']=statistics.stdev(values)
    result['parameters']=selected[0]['parameters']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--num-threads',type=int,default=2)
    parser.add_argument('--smoke',action='store_true',help='One epoch per candidate; no model selection or repeats')
    args=parser.parse_args()
    if args.num_threads<1: parser.error('num-threads must be positive')
    if args.device=='cuda' and not torch.cuda.is_available(): parser.error('CUDA unavailable')
    torch.set_num_threads(args.num_threads)
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    rows=[]; configs=[]
    for path in CANDIDATE_PATHS:
        config=load_config(path,output,1 if args.smoke else None)
        configs.append(config); rows.append(execute(config,args.device,args.num_threads))
        save_table(output/'comparison.csv',rows); write_json(output/'comparison.json',rows)
    if args.smoke:
        print('SMOKE CHECKS PASSED; these scores are not full experiment results.'); return
    winner=select_candidate(rows,[c['experiment_name'] for c in configs])
    selected=next(c for c in configs if c['experiment_name']==winner['experiment'])
    # Strip paths to keep the selection stable when Colab extracts to a new folder.
    selected_portable=json.loads(json.dumps(selected)); selected_portable.update(data_dir='lab2_EEG_classification/data',output_dir='outputs/bonus')
    decision={'criterion':'validation_accuracy; ties prefer ELU 0.001, ELU 0.0005, then LeakyReLU 0.001',
              'selected_experiment':winner['experiment'],'selected_config':selected_portable,
              'candidate_validation_accuracies':{r['experiment']:r['validation_accuracy'] for r in rows},
              'note':'Frozen before seeds 43/44; test scores are not used for selection.'}
    path=output/'selection.json'
    if path.exists() and json.loads(path.read_text())!=decision: raise RuntimeError('Stored selection differs; inspect protocol before continuing')
    write_json(path,decision); print('Selected using validation:',winner['experiment'],flush=True)
    for seed in [43,44]:
        config=json.loads(json.dumps(selected)); config['seed']=seed
        config['experiment_name']=selected['experiment_name'].removesuffix('_seed42')+f'_seed{seed}'
        rows.append(execute(config,args.device,args.num_threads))
        save_table(output/'comparison.csv',rows); write_json(output/'comparison.json',rows)
    selected_stats=summarize_selected(rows,selected['experiment_name'])
    comparisons=[]
    with (ROOT/'results/seed_summary.csv').open() as stream:
        for row in csv.DictReader(stream):
            item={'model':'EEGNet','configuration':row['configuration'],'n':int(row['n']),
                  'model_seeds':row['model_seeds'],'split_seed':int(row['split_seed'])}
            for metric in ['validation_accuracy','selected_test_accuracy']:
                for suffix in ['mean','sample_std']: item[metric+'_'+suffix]=float(row[metric+'_'+suffix])
            item['parameters']=17874; comparisons.append(item)
    comparisons.append(selected_stats)
    save_table(output/'seed_summary.csv',comparisons); write_json(output/'seed_summary.json',comparisons)
    from bonus_results import create_figures
    create_figures(output)
    write_json(output/'complete.json',{'status':'gpu_batch_complete_pending_independent_review',
               'selected_experiment':winner['experiment'],'new_full_runs':len(rows),'split_seed':42,
               'sample_std_ddof':1,'eegnet_retrained':False})
    print('BONUS GPU BATCH COMPLETE. Return Lab2_bonus_results.zip for independent review.')
    print((output/'seed_summary.csv').read_text())


if __name__=='__main__': main()
