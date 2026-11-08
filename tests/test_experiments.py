import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('experiments',ROOT/'scripts/run_experiments.py')
experiments=importlib.util.module_from_spec(spec); spec.loader.exec_module(experiments)


class ExperimentTests(unittest.TestCase):
    def test_nine_unique_one_factor_comparisons(self):
        baseline=json.loads((ROOT/'configs/baseline.json').read_text())
        configs=[json.loads(p.read_text()) for p in (ROOT/'configs/experiments').glob('*.json')]
        self.assertEqual(len(configs),9)
        changes=[]
        for config in configs:
            self.assertEqual(config['seed'],42)
            self.assertEqual(config['evaluation'],baseline['evaluation'])
            changed=[(section,key,value) for section in ['model','training'] for key,value in config[section].items() if value!=baseline[section][key]]
            self.assertEqual(len(changed),1); changes.append(str(changed))
        self.assertEqual(len(set(changes)),9)

    def test_identity_keeps_epochs_and_split(self):
        config=json.loads((ROOT/'configs/baseline.json').read_text())
        moved=json.loads(json.dumps(config)); moved.update(data_dir='/other/data',output_dir='/other/output')
        self.assertEqual(experiments.identity(config),experiments.identity(moved))
        moved['training']['epochs']=1
        self.assertNotEqual(experiments.identity(config),experiments.identity(moved))

    def test_real_batch_reuses_run_and_preserves_verification(self):
        from test_pipeline import fixture_data
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); data=root/'data'; data.mkdir(); fixture_data(data)
            config=json.loads((ROOT/'configs/selected.json').read_text()); config['data_dir']=str(data)
            path=root/'config.json'; path.write_text(json.dumps(config)); output=root/'runs'
            command=[sys.executable,str(ROOT/'scripts/run_experiments.py'),'--configs',str(path),'--epochs','1','--device','cpu','--output-dir',str(output)]
            first=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            folders=list(output.glob('*/summary.json')); self.assertEqual(len(folders),1)
            verification=folders[0].parent/'verification.json'; original=verification.read_bytes()
            second=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(second.returncode,0,second.stderr)
            self.assertEqual(len(list(output.glob('*/summary.json'))),1)
            self.assertEqual(verification.read_bytes(),original)
            self.assertTrue((folders[0].parent/'local_review.json').exists())
