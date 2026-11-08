import argparse
import csv
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lab2_EEG_classification'))
sys.path.insert(0, str(ROOT / 'scripts'))
import torch
from main import load_config, run, write_json
from verify_run import verify_run


def identity(config):
    return {k:v for k,v in config.items() if k not in ('data_dir','output_dir')}


def verify_saved(folder, data_dir, device):

    path=folder/'verification.json'
    original=path.read_bytes() if path.exists() else None
    try:
        result=verify_run(folder,data_dir,device,device=='cuda')
        if original is not None:
            write_json(folder/'local_review.json',result)
        return result
    finally:
        if original is not None:
            path.write_bytes(original)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs',type=Path,nargs='+',required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/experiments')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--seeds',type=int,nargs='+',help='Model seeds; split seed stays fixed')
    parser.add_argument('--epochs',type=int,help='Override only for smoke checks')
    parser.add_argument('--num-threads',type=int,default=2)
    args=parser.parse_args()
    if args.num_threads<1: parser.error('--num-threads must be positive')
    if args.device=='cuda' and not torch.cuda.is_available(): parser.error('CUDA unavailable')
    torch.set_num_threads(args.num_threads)
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=True)
    rows=[]; requested=set()
    for path in args.configs:
        for seed in args.seeds or [None]:
            config=load_config(SimpleNamespace(config=path,epochs=args.epochs,batch_size=None,
                               lr=None,seed=seed,output_dir=output))
            if seed is not None:
                config['experiment_name']=re.sub(r'_seed\d+$','',config['experiment_name'])+f'_seed{seed}'
            key=json.dumps(identity(config),sort_keys=True)
            if key in requested: continue
            requested.add(key)
            matches=[]
            for saved in output.glob(config['experiment_name']+'_*/config.json'):
                if identity(json.loads(saved.read_text()))==identity(config) and (saved.parent/'summary.json').exists():
                    matches.append(saved.parent)
            if len(matches)>1: raise RuntimeError(f'Multiple matching completed runs: {matches}')
            folder=matches[0] if matches else run(config,torch.device(args.device),args.num_threads)
            verify_saved(folder,Path(config['data_dir']),args.device)
            summary=json.loads((folder/'summary.json').read_text())
            row={'experiment':config['experiment_name'],'run_dir':str(folder),'seed':config['seed'],
                 'epochs':config['training']['epochs'],'selected_epoch':summary['selected_epoch'],
                 'validation_accuracy':summary['selected_checkpoint_scores']['validation']['accuracy'],
                 'selected_test_accuracy':summary['selected_checkpoint_scores']['test']['accuracy'],
                 'peak_monitored_test_accuracy':summary['peak_test_accuracy']}
            rows.append(row)
            write_json(output/'comparison.json',rows)
            with (output/'comparison.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader(); writer.writerows(rows)
            print('Verified:',config['experiment_name'],flush=True)
    print('Saved comparison:',output/'comparison.csv')


if __name__=='__main__': main()
