import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab2_EEG_classification"))
sys.path.insert(0, str(ROOT / "scripts"))
import dataloader
import main as training
from models.EEGNet import EEGNet
from verify_run import verify_run


def base_config():
    return json.loads((ROOT / "configs/baseline.json").read_text())


def fixture_data(directory):
    rng = np.random.default_rng(10)
    for subject in ("S4b", "X11b"):
        for split, n in (("train", 20), ("test", 8)):
            labels = np.tile([1, 2], n // 2)
            signal = rng.normal(0, 0.3, (n, 750, 2)).astype(np.float32)
            signal += (labels[:, None, None] - 1.5).astype(np.float32)
            signal[0, 0, 0] = np.nan
            np.savez(directory / f"{subject}_{split}.npz", signal=signal, label=labels)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_architectures_and_activations_have_finite_gradients(self):
        for mode in ("slide", "depthwise_separable"):
            for activation in ("elu", "relu", "leaky_relu"):
                with self.subTest(mode=mode, activation=activation):
                    model = EEGNet(activation=activation, temporal_block=mode)
                    logits = model(torch.randn(3, 1, 2, 750))
                    self.assertEqual(tuple(logits.shape), (3, 2))
                    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in model.parameters()))
        self.assertEqual(sum(p.numel() for p in EEGNet().parameters()), 17874)

    def test_invalid_model_options_and_input(self):
        for options in ({"dropout": 1}, {"dropout": float('nan')},
                        {"elu_alpha": float('nan')}, {"elu_alpha": float('inf')},
                        {"negative_slope": float('nan')}, {"activation": "unknown"},
                        {"temporal_block": "unknown"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                EEGNet(**options)
        with self.assertRaises(ValueError):
            EEGNet()(torch.zeros(2, 1, 2, 749))

    def test_evaluation_does_not_update_weights_or_batchnorm(self):
        model = EEGNet()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        loader = DataLoader(TensorDataset(torch.randn(3, 1, 2, 750), torch.tensor([0, 1, 0])), batch_size=2)
        training.epoch_metrics(model, loader, torch.nn.CrossEntropyLoss(), torch.device('cpu'))
        self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_weighted_loss_on_partial_batch(self):
        logits = torch.tensor([[3., 0.], [0., 3.], [1., 0.], [2., 0.], [0., 1.]])
        labels = torch.tensor([0, 1, 1, 0, 0])
        loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
        metrics = training.epoch_metrics(torch.nn.Identity(), loader, torch.nn.CrossEntropyLoss(), torch.device('cpu'))
        self.assertAlmostEqual(metrics['loss'], torch.nn.functional.cross_entropy(logits, labels).item(), places=6)
        self.assertEqual(metrics['accuracy'], 60.)

    def test_nonfinite_loss_fails(self):
        loader = DataLoader(TensorDataset(torch.full((2, 2), float('nan')), torch.tensor([0, 1])))
        with self.assertRaises(FloatingPointError):
            training.epoch_metrics(torch.nn.Identity(), loader, torch.nn.CrossEntropyLoss(), torch.device('cpu'))


class DataTests(unittest.TestCase):
    def test_split_is_disjoint_stratified_and_repeatable(self):
        labels = np.tile([0, 1], 20)
        subjects = np.repeat([0, 1], 20)
        a, b = dataloader.split_indices(labels, subjects)
        c, d = dataloader.split_indices(labels, subjects)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(set(a) | set(b), set(range(40)))
        for group in range(4):
            self.assertEqual(int(((subjects * 2 + labels)[b] == group).sum()), 2)
        with self.assertRaises(ValueError):
            dataloader.split_indices(labels, subjects, fraction=0)

    def test_imputation_ignores_validation_and_test_values(self):
        labels = np.tile([0, 1], 20)
        subjects = np.repeat([0, 1], 20)
        train, val = dataloader.split_indices(labels, subjects)
        x = np.full((40, 1, 2, 750), 2., dtype=np.float32)
        x[val] = 10000.
        x[train[0], 0, 0, 0] = np.nan
        x[val[0], 0, 0, 0] = np.nan
        test = np.full((4, 1, 2, 750), -9999., dtype=np.float32)
        test[0, 0, 0, 0] = np.nan
        with patch.object(dataloader, 'load_split', side_effect=[(x, labels, subjects), (test, labels[:4], subjects[:4])]):
            splits, metadata = dataloader.prepare_data()
        self.assertEqual(metadata['fill_value'], 2.)
        for name in ('train', 'validation', 'test'):
            self.assertEqual(splits[name][0][0, 0, 0, 0], 2.)
            self.assertTrue(np.isfinite(splits[name][0]).all())
        self.assertEqual(splits['validation'][0][0, 0, 0, 1], 10000.)

    def test_bad_labels_and_infinity_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            fixture_data(directory)
            p = directory / 'S4b_train.npz'
            with np.load(p) as data:
                signal, labels = data['signal'].copy(), data['label'].copy()
            labels[0] = 3
            np.savez(p, signal=signal, label=labels)
            with self.assertRaisesRegex(ValueError, 'labels'):
                dataloader.load_split(directory, 'train')
            labels[0] = 1
            signal[1, 0, 0] = np.inf
            np.savez(p, signal=signal, label=labels)
            with self.assertRaisesRegex(ValueError, 'infinite'):
                dataloader.load_split(directory, 'train')


class ConfigTests(unittest.TestCase):
    def parse(self, config, **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            path.write_text(json.dumps(config))
            args = argparse.Namespace(config=path, epochs=None, batch_size=None, lr=None,
                                      seed=None, output_dir=None, **{})
            for key, value in overrides.items():
                setattr(args, key, value)
            return training.load_config(args)

    def test_model_seed_override_keeps_split_seed(self):
        config = self.parse(base_config(), seed=99, epochs=2)
        self.assertEqual(config['seed'], 99)
        self.assertEqual(config['evaluation']['split_seed'], 42)
        self.assertEqual(config['training']['epochs'], 2)
        self.assertTrue(Path(config['data_dir']).is_absolute())

    def test_invalid_config_fails(self):
        for section, key, value in [('training', 'epochs', 0), ('training', 'batch_size', True),
                                    ('training', 'learning_rate', float('nan')),
                                    ('evaluation', 'validation_fraction', 1),
                                    ('evaluation', 'checkpoint_metric', 'test_accuracy')]:
            config = base_config()
            config[section][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.parse(config)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.tmp.name)
        cls.data = cls.directory / 'data'
        cls.data.mkdir()
        fixture_data(cls.data)
        cls.config = base_config()
        cls.config['data_dir'] = str(cls.data)
        cls.config['output_dir'] = str(cls.directory / 'runs')
        cls.config['training'].update(epochs=3, batch_size=7)
        cls.first = training.run(copy.deepcopy(cls.config), torch.device('cpu'), 2)
        cls.repeat = training.run(copy.deepcopy(cls.config), torch.device('cpu'), 2)
        no_test = copy.deepcopy(cls.config)
        no_test['evaluation']['record_test_curves'] = False
        cls.no_test = training.run(no_test, torch.device('cpu'), 2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_run_artifacts_checkpoint_selection_and_reload(self):
        result = verify_run(self.first)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['epochs'], 3)
        self.assertNotEqual(self.first, self.repeat)

    def test_repeated_run_is_exactly_reproducible_on_cpu(self):
        self.assertEqual((self.first / 'metrics.csv').read_bytes(), (self.repeat / 'metrics.csv').read_bytes())
        for checkpoint in ('best.pt', 'last.pt'):
            a = torch.load(self.first / checkpoint, weights_only=True)['model_state_dict']
            b = torch.load(self.repeat / checkpoint, weights_only=True)['model_state_dict']
            self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_test_monitoring_does_not_change_training(self):
        with (self.first / 'metrics.csv').open() as f:
            monitored = list(csv.DictReader(f))
        with (self.no_test / 'metrics.csv').open() as f:
            unmonitored = list(csv.DictReader(f))
        for a, b in zip(monitored, unmonitored):
            for key in ('train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy'):
                self.assertEqual(a[key], b[key])
            self.assertEqual(b['test_loss'], '')
        self.assertEqual(verify_run(self.no_test)['status'], 'passed')

    def test_verifier_rejects_corrupted_summary(self):
        path = self.first / 'summary.json'
        original = path.read_text()
        changed = json.loads(original)
        changed['selected_epoch'] = 99
        path.write_text(json.dumps(changed))
        try:
            with self.assertRaisesRegex(ValueError, 'validation-selected'):
                verify_run(self.first)
        finally:
            path.write_text(original)

    def test_cli_from_other_directory_and_invalid_epochs(self):
        result = subprocess.run([sys.executable, str(ROOT / 'lab2_EEG_classification/main.py'),
                                 '--epochs', '0'], cwd=self.directory, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('epochs must be a positive integer', result.stderr)

    def test_cli_rejects_cuda_when_unavailable(self):
        if torch.cuda.is_available():
            self.skipTest('CUDA is available in this environment')
        result = subprocess.run([sys.executable, str(ROOT / 'lab2_EEG_classification/main.py'),
                                 '--device', 'cuda', '--epochs', '1'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('CUDA was requested but is unavailable', result.stderr)


if __name__ == '__main__':
    unittest.main()
