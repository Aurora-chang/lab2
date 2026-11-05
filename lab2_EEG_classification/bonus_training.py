"""DeepConvNet training using the original EEGNet data/evaluation utilities.

The historical main.py is intentionally unchanged to preserve EEGNet hashes.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
import platform
import random
import sys
import time

import matplotlib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from dataloader import prepare_data
from main import (PROJECT_ROOT,FIELDS,epoch_metrics,plot_curves,save_checkpoint,
                  source_hashes as eegnet_source_hashes,write_json)
from models.DeepConvNet import DeepConvNet


def source_hashes():
    hashes=eegnet_source_hashes()
    for path in [Path(__file__),PROJECT_ROOT/'lab2_EEG_classification/models/DeepConvNet.py']:
        hashes[str(path.relative_to(PROJECT_ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def load_config(path, output_dir=None, epochs=None, seed=None):
    config=json.loads(Path(path).read_text())
    if config['model']['name']!='DeepConvNet': raise ValueError('Expected DeepConvNet configuration')
    if epochs is not None: config['training']['epochs']=epochs
    if seed is not None: config['seed']=seed
    if output_dir is not None: config['output_dir']=str(output_dir)
    for key in ['data_dir','output_dir']:
        value=Path(config[key]).expanduser()
        config[key]=str((PROJECT_ROOT/value).resolve() if not value.is_absolute() else value.resolve())
    t,e=config['training'],config['evaluation']
    for key in ['epochs','batch_size','artifact_interval']:
        if type(t[key]) is not int or t[key]<1: raise ValueError(f'Invalid {key}')
    if t['optimizer']!='adam' or t['loss']!='cross_entropy': raise ValueError('Expected Adam/cross-entropy')
    for key in ['learning_rate','weight_decay']:
        if not math.isfinite(t[key]) or t[key]<0: raise ValueError(f'Invalid {key}')
    if t['learning_rate']==0: raise ValueError('Learning rate must be positive')
    if not math.isfinite(e['validation_fraction']) or not 0<e['validation_fraction']<1:
        raise ValueError('Invalid validation fraction')
    if e['checkpoint_metric']!='validation_accuracy' or type(e['stratify']) is not bool or type(e['record_test_curves']) is not bool:
        raise ValueError('Invalid evaluation policy')
    for value in [config['seed'],e['split_seed']]:
        if type(value) is not int or not 0<=value<2**32: raise ValueError('Invalid seed')
    name=config['experiment_name']
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('Invalid experiment name')
    # Constructor validates activation, dropout, and constraints before any output.
    with torch.random.fork_rng(devices=[]):
        DeepConvNet(**{k:v for k,v in config['model'].items() if k!='name'})
    return config


class ConstrainedAdam(torch.optim.Adam):
    def __init__(self,model,**kwargs):
        self.model=model
        super().__init__(model.parameters(),**kwargs)

    def step(self,closure=None):
        result=super().step(closure)
        self.model.constrain_weights()
        return result


def evaluate_with_confusion(model,loader,device):
    model.eval(); total=0.; matrix=np.zeros((2,2),dtype=np.int64)
    with torch.inference_mode():
        for inputs,labels in loader:
            logits=model(inputs.to(device)); targets=labels.to(device)
            total+=nn.functional.cross_entropy(logits,targets,reduction='sum').item()
            np.add.at(matrix,(labels.numpy(),logits.argmax(1).cpu().numpy()),1)
    return {'loss':total/int(matrix.sum()),'accuracy':100.*float(np.trace(matrix))/int(matrix.sum())},matrix.tolist()


def run(config,device,num_threads=2):
    import csv
    torch.set_num_threads(num_threads)
    seed=config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    e,t=config['evaluation'],config['training']
    splits,prep=prepare_data(config['data_dir'],e['validation_fraction'],e['split_seed'],e['stratify'])
    model=DeepConvNet(**{k:v for k,v in config['model'].items() if k!='name'}).to(device)
    optimizer=ConstrainedAdam(model,lr=t['learning_rate'],weight_decay=t['weight_decay'])
    loaders={name:DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=t['batch_size'],
                            shuffle=name=='train',generator=torch.Generator().manual_seed(seed if name=='train' else seed+1),
                            num_workers=0,pin_memory=device.type=='cuda') for name,(x,y) in splits.items()}
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder=Path(config['output_dir'])/(config['experiment_name']+'_'+stamp); folder.mkdir(parents=True,exist_ok=False)
    logger=logging.getLogger('bonus.'+stamp); logger.setLevel(logging.INFO); logger.propagate=False
    for handler in [logging.FileHandler(folder/'training.log'),logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s')); logger.addHandler(handler)
    try:
        write_json(folder/'config.json',config); write_json(folder/'preprocessing.json',prep)
        write_json(folder/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
                   'torch':str(torch.__version__),'numpy':np.__version__,'matplotlib':matplotlib.__version__,
                   'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                   'cuda':torch.version.cuda,'threads':torch.get_num_threads(),'source_sha256':source_hashes(),
                   'dataset_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(config['data_dir']).glob('*.npz'))}})
        (folder/'model.txt').write_text(str(model)+'\n')
        rows=[]; best=-1.; selected=None; peak=None; peak_epoch=None; started=time.perf_counter()
        logger.info('DeepConvNet | device %s | parameters %d | results %s',device,sum(p.numel() for p in model.parameters()),folder)
        with (folder/'metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS); writer.writeheader()
            for epoch in range(1,t['epochs']+1):
                row={'epoch':epoch,'learning_rate':t['learning_rate']}
                for split in ['train','validation','test']:
                    scores=epoch_metrics(model,loaders[split],nn.CrossEntropyLoss(),device,optimizer if split=='train' else None) if split!='test' or e['record_test_curves'] else {'loss':None,'accuracy':None}
                    row.update({f'{split}_{key}':value for key,value in scores.items()})
                rows.append(row); writer.writerow(row); stream.flush()
                if row['test_accuracy'] is not None and (peak is None or row['test_accuracy']>peak): peak,peak_epoch=row['test_accuracy'],epoch
                improved=row['validation_accuracy']>best
                if improved: best,selected=row['validation_accuracy'],epoch
                checkpoint={'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),
                            'config':config,'preprocessing':prep,'metrics':row,'best_validation_accuracy':best,'best_epoch':selected}
                if improved: save_checkpoint(folder/'best.pt',checkpoint)
                if epoch%t['artifact_interval']==0 or epoch==t['epochs']:
                    save_checkpoint(folder/'last.pt',checkpoint); plot_curves(rows,folder/'curves.png')
                logger.info('Epoch %d/%d | train %.2f%% | validation %.2f%% | test %s',epoch,t['epochs'],row['train_accuracy'],row['validation_accuracy'],row['test_accuracy'])
        model.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=True)['model_state_dict'])
        selected_scores={}; matrices={}
        for split,loader in loaders.items(): selected_scores[split],matrices[split]=evaluate_with_confusion(model,loader,device)
        write_json(folder/'confusion_matrices.json',{'selected_epoch':selected,'axes':'rows=true, columns=predicted','label_mapping':{'0':'original label 1','1':'original label 2'},'splits':matrices})
        summary={'status':'complete','selection_criterion':'validation_accuracy','selected_epoch':selected,
                 'best_validation_accuracy':best,'selected_checkpoint_scores':selected_scores,
                 'peak_test_accuracy':peak,'peak_test_epoch':peak_epoch,'accuracy_unit':'percent',
                 'peak_test_note':'Maximum over monitored epochs; not an independent held-out estimate.',
                 'duration_seconds':time.perf_counter()-started,'parameters':sum(p.numel() for p in model.parameters())}
        write_json(folder/'summary.json',summary)
        logger.info('Selected epoch %d | validation %.2f%% | selected test %.2f%% | monitored test max %s',selected,best,selected_scores['test']['accuracy'],peak)
        return folder
    except BaseException:
        logger.exception('Run incomplete; partial artifacts retained'); raise
    finally:
        for handler in list(logger.handlers): handler.close(); logger.removeHandler(handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True); parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--epochs',type=int); parser.add_argument('--seed',type=int)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda'); parser.add_argument('--num-threads',type=int,default=2)
    args=parser.parse_args()
    if args.num_threads<1: parser.error('num-threads must be positive')
    if args.device=='cuda' and not torch.cuda.is_available(): parser.error('CUDA unavailable')
    run(load_config(args.config,args.output_dir,args.epochs,args.seed),torch.device(args.device),args.num_threads)


if __name__=='__main__': main()
