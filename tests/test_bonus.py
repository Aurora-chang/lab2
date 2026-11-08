import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'lab2_EEG_classification'))
sys.path.insert(0,str(ROOT/'scripts'))
from models.DeepConvNet import DeepConvNet
from bonus_training import ConstrainedAdam,load_config,run
from run_bonus import select_candidate,summarize_selected
from verify_run import verify_run
from test_pipeline import fixture_data


class BonusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_forward_backward_and_shape(self):
        for activation in ['elu','leaky_relu','relu']:
            model=DeepConvNet(activation=activation)
            logits=model(torch.randn(3,1,2,750))
            self.assertEqual(tuple(logits.shape),(3,2))
            torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,0])).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertEqual(sum(p.numel() for p in model.parameters()),266352)
        with self.assertRaises(ValueError): model(torch.randn(2,1,2,500))

    def test_constraints_applied_after_optimizer_step(self):
        model=DeepConvNet(); optimizer=ConstrainedAdam(model,lr=1.)
        loss=model(torch.randn(3,1,2,750)).square().sum(); loss.backward(); optimizer.step()
        for layer in model.modules():
            if isinstance(layer,(torch.nn.Conv2d,torch.nn.Linear)):
                bound=.5 if isinstance(layer,torch.nn.Linear) else 2.
                self.assertTrue(bool((layer.weight.flatten(1).norm(2,dim=1)<=bound+1e-5).all()))

    def test_invalid_parameters(self):
        for params in [{'dropout':float('nan')},{'dropout':1},{'elu_alpha':0},{'conv_max_norm':float('inf')},{'classifier_max_norm':0},{'activation':'unknown'}]:
            with self.subTest(params=params),self.assertRaises(ValueError): DeepConvNet(**params)

    def test_validation_only_selection_and_sample_std(self):
        rows=[{'experiment':'a','validation_accuracy':88,'selected_test_accuracy':70},
              {'experiment':'b','validation_accuracy':87,'selected_test_accuracy':99}]
        self.assertEqual(select_candidate(rows,['a','b'])['experiment'],'a')
        rows[1]['validation_accuracy']=88
        self.assertEqual(select_candidate(rows,['a','b'])['experiment'],'a')
        rows=[{'experiment':f'a_seed{seed}','seed':seed,'validation_accuracy':score,
               'selected_test_accuracy':score,'parameters':266352} for seed,score in zip([42,43,44],[80,82,84])]
        stat=summarize_selected(rows,'a_seed42')
        self.assertEqual(stat['selected_test_accuracy_mean'],82)
        self.assertEqual(stat['selected_test_accuracy_sample_std'],2)

    def test_training_reload_confusion_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary); data=folder/'data'; data.mkdir(); fixture_data(data)
            config=load_config(ROOT/'configs/bonus/deepconvnet_elu.json',folder/'runs',epochs=2)
            config['data_dir']=str(data); config['training']['batch_size']=8

            config['training']['artifact_interval']=25
            target=run(config,torch.device('cpu'),2)
            result=verify_run(target,data,'cpu')
            self.assertEqual(result['status'],'passed')
            self.assertIn('max_norm_constraints',result['checks'])
            self.assertIn('all_split_confusion_matrices',result['checks'])
            summary=json.loads((target/'summary.json').read_text())
            matrices=json.loads((target/'confusion_matrices.json').read_text())
            for split,size in [('train',32),('validation',8),('test',16)]:
                matrix=np.asarray(matrices['splits'][split])
                self.assertEqual(matrix.sum(),size)
                self.assertAlmostEqual(100*np.trace(matrix)/size,summary['selected_checkpoint_scores'][split]['accuracy'])
            matrices['splits']['test'][0][0]+=1
            (target/'confusion_matrices.json').write_text(json.dumps(matrices))
            with self.assertRaisesRegex(ValueError,'Confusion counts differ'): verify_run(target,data,'cpu')

    def test_full_batch_selects_before_repeats_and_reuses_eegnet(self):
        import run_bonus
        calls=[]
        def fake_execute(config,device,threads):
            calls.append(config['experiment_name'])
            score=90 if config['training']['learning_rate']==0.0005 else 85
            return {'experiment':config['experiment_name'],'seed':config['seed'],'split_seed':42,
                    'epochs':150,'activation':config['model']['activation'],
                    'learning_rate':config['training']['learning_rate'],'dropout':.5,
                    'selected_epoch':5,'validation_accuracy':score,'selected_test_accuracy':80,
                    'peak_monitored_test_accuracy':82,'peak_test_epoch':6,'parameters':266352,
                    'duration_seconds':1,'run_dir':'simulated'}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(run_bonus,'execute',side_effect=fake_execute), patch('bonus_results.create_figures'), patch.object(sys,'argv',['run_bonus','--output-dir',directory,'--device','cpu']):
                run_bonus.main()
            folder=Path(directory)
            selection=json.loads((folder/'selection.json').read_text())
            self.assertEqual(selection['selected_experiment'],'deepconvnet_elu_lr0005_seed42')
            self.assertEqual(len(calls),5)
            self.assertEqual(calls[-2:],['deepconvnet_elu_lr0005_seed43','deepconvnet_elu_lr0005_seed44'])
            complete=json.loads((folder/'complete.json').read_text())
            self.assertFalse(complete['eegnet_retrained'])
            self.assertEqual(complete['new_full_runs'],5)

    def test_public_bonus_snapshots_preserve_provenance_and_metrics(self):
        import csv
        import hashlib
        folders=list((ROOT/'results/bonus/runs').iterdir())
        self.assertEqual(len(folders),5)
        for folder in folders:
            environment=json.loads((folder/'environment.json').read_text())
            for name,digest in environment['source_sha256'].items():
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest)
            proof=json.loads((folder/'verification.json').read_text())
            self.assertEqual(proof['gpu_verification']['status'],'passed')
            self.assertEqual(proof['independent_cpu_verification']['status'],'passed')
            with (folder/'metrics.csv').open() as stream: metrics=list(csv.DictReader(stream))
            self.assertEqual(len(metrics),150)
            summary=json.loads((folder/'summary.json').read_text())
            chosen=max(metrics,key=lambda r:float(r['validation_accuracy']))
            self.assertEqual(int(chosen['epoch']),summary['selected_epoch'])
            matrices=json.loads((folder/'confusion_matrices.json').read_text())
            for split,size in [('train',864),('validation',216),('test',1080)]:
                matrix=np.asarray(matrices['splits'][split])
                self.assertEqual(matrix.sum(),size)
                self.assertAlmostEqual(100*np.trace(matrix)/size,summary['selected_checkpoint_scores'][split]['accuracy'])

    def test_public_bonus_selection_and_seed_statistics(self):
        output=ROOT/'results/bonus'
        rows=json.loads((output/'comparison.json').read_text())
        candidates=[r['experiment'] for r in rows if r['seed']==42]
        selection=json.loads((output/'selection.json').read_text())
        self.assertEqual(select_candidate(rows,candidates)['experiment'],selection['selected_experiment'])
        self.assertEqual(summarize_selected(rows,selection['selected_experiment']),json.loads((output/'seed_summary.json').read_text())[-1])
        for path in output.rglob('*'):
            if path.is_file():
                self.assertNotIn(path.suffix,['.pt','.npz','.zip'])
                if path.suffix in ['.json','.csv']:
                    self.assertNotIn('/Users/',path.read_text())
                    self.assertNotIn('/content/drive/',path.read_text())
