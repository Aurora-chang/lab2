import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('analysis',ROOT/'scripts/analyze_results.py')
analysis=importlib.util.module_from_spec(spec); spec.loader.exec_module(analysis)


class ResultsTests(unittest.TestCase):
    def test_all_17_snapshots_and_source_hashes(self):
        runs=analysis.load_runs()
        self.assertEqual(len(runs),17)
        for run in runs:
            env=json.loads((run['folder']/'environment.json').read_text())
            for path,digest in env['source_sha256'].items():
                self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),digest)
            proof=json.loads((run['folder']/'verification.json').read_text())
            self.assertEqual(proof['gpu_verification']['status'],'passed')
            self.assertEqual(proof['independent_cpu_verification']['status'],'passed')

    def test_seed_summary_matches_per_run_scores(self):
        with (ROOT/'results/seed_summary.csv').open() as stream: stats=list(csv.DictReader(stream))
        runs=analysis.load_runs()
        for family,prefix in [('baseline','elu_baseline_seed'),('leaky_relu','leaky_relu_seed')]:
            chosen=[r for r in runs if r['folder'].name.startswith(prefix)]
            self.assertEqual(sorted(r['config']['seed'] for r in chosen),[42,43,44])
            row=next(r for r in stats if r['configuration']==family)
            values=[r['summary']['selected_checkpoint_scores']['test']['accuracy'] for r in chosen]
            self.assertAlmostEqual(sum(values)/3,float(row['selected_test_accuracy_mean']))
            self.assertAlmostEqual(analysis.statistics.stdev(values),float(row['selected_test_accuracy_sample_std']))

    def test_confusion_counts_reproduce_all_selected_scores(self):
        saved=json.loads((ROOT/'results/confusion_matrices.json').read_text())
        summary=json.loads((ROOT/'results/runs/leaky_relu_seed42/summary.json').read_text())
        for split,size in [('train',864),('validation',216),('test',1080)]:
            matrix=saved['splits'][split]['counts']
            self.assertEqual(sum(map(sum,matrix)),size)
            self.assertAlmostEqual(100*(matrix[0][0]+matrix[1][1])/size,summary['selected_checkpoint_scores'][split]['accuracy'])

    def test_no_machine_specific_paths_in_public_snapshots(self):
        for path in (ROOT/'results').rglob('*.json'):
            self.assertNotIn('/Users/',path.read_text())
            self.assertNotIn('/content/drive/',path.read_text())
