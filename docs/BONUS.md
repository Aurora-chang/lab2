# DeepConvNet Bonus

Implementation, all five full GPU experiments, and independent local verification are complete. Selected DeepConvNet ELU achieved **86.57% ± 0.46% validation** and **82.96% ± 0.76% selected-checkpoint test accuracy** across seeds 42–44. See the [verified bonus results](../results/bonus/README.md). This document records the implementation and reproduction protocol.

## Optional Reproduction on Colab

1. Open `notebooks/Colab.ipynb` in a GPU runtime.
2. Run sections **1 and 2** using the latest `outputs/Lab2_colab.zip`.
3. Skip EEGNet sections 3–5; run **section 6**.
4. Export **`Lab2_bonus_results.zip`** for independent checkpoint and statistics verification. The existing reviewed results do not need to be trained again.

Use the new ZIP; the earlier project upload does not contain DeepConvNet. Results are saved to `MyDrive/Lab2/outputs/bonus_deepconvnet/`. A separate smoke directory prevents short checks from being mixed with full experiments. On restart, repeat setup and section 6: matching completed runs are reverified and reused; incomplete runs restart at epoch 1. Do not run multiple batches against the same output directory.

## Architecture and implementation

Independent PyTorch implementation: [`models/DeepConvNet.py`](../lab2_EEG_classification/models/DeepConvNet.py), based on the four-block architecture in [Schirrmeister et al. (2017), *Deep learning with convolutional neural networks for EEG decoding and visualization*](https://arxiv.org/abs/1703.05051). The [authors' Braindecode Deep4Net documentation](https://braindecode.org/stable/generated/braindecode.models.Deep4Net.html) provides an architecture reference.

| Block | Operations | Output shape excluding batch |
| --- | --- | --- |
| 1 | Temporal Conv 1→25 `(1,10)`; spatial Conv 25→25 `(2,1)`; BN; activation; MaxPool `(1,3)` stride 3; dropout | `(25,1,247)` |
| 2 | Conv 25→50 `(1,10)`; BN; activation; MaxPool 3; dropout | `(50,1,79)` |
| 3 | Conv 50→100 `(1,10)`; BN; activation; MaxPool 3; dropout | `(100,1,23)` |
| 4 | Conv 100→200 `(1,10)`; BN; activation; MaxPool 3; dropout | `(200,1,4)` |
| Classifier | Flatten 800 → Linear 2 raw logits | `(2,)` |

Input is `(B,1,2,750)`, with **266,352** trainable parameters. Temporal convolutions have no padding. Convolutions use no bias and are followed by batch normalization after each complete block. BatchNorm uses epsilon 1e-5 and PyTorch momentum 0.1. We initialize convolution/dense weights with Xavier uniform and enforce per-output-filter max norm 2.0 for convolutions and 0.5 for the classifier after each Adam update.

Max-norm regularization is also described in the [ARL EEGModels DeepConvNet reference](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py). That reference uses shorter kernels/pooling for a different sampling rate; this implementation uses kernel 10/pool 3, a two-class dense head, and trial-wise training. It is an adaptation, not an exact reproduction of the paper's original preprocessing, cropped training, or stopping protocol. No external model library is required.

## Fixed experiment protocol

The original EEGNet data loader and training helpers are reused. **The historical EEGNet `main.py`, `dataloader.py`, and `EEGNet.py` are unchanged**, preserving their original source hashes. A dedicated `bonus_training.py` handles the new architecture. The verifier dispatches by model type.

All full runs use the same 864/216 train/validation split, 1,080 test trials, split seed 42, training-only scalar NaN imputation, no normalization, batch size 64, Adam weight decay 0.001, and 150 epochs. Test curves are recorded for discussion but do not participate in model or checkpoint selection.

Three candidates at model seed 42:

| Candidate | Activation | Learning rate | Dropout |
| --- | --- | ---: | ---: |
| DeepConvNet reference | ELU alpha 1 | 0.001 | 0.5 |
| Lower learning rate | ELU alpha 1 | 0.0005 | 0.5 |
| Activation comparison | LeakyReLU slope 0.01 | 0.001 | 0.5 |

Select the greatest validation-selected checkpoint validation accuracy. Ties prefer the candidate order above. Freeze the decision in `selection.json` **before** running the selected configuration with model seeds 43 and 44. This produces **five new full runs**, with no EEGNet retraining.

Final comparisons reuse the original EEGNet ELU and selected LeakyReLU means from `results/seed_summary.csv`. DeepConvNet uses its own tuned learning rate/dropout/max-norm settings, so the comparison evaluates practical model configurations, not architecture alone. Seed 42 is used for tuning and contributes to the means. Sample SD uses ddof=1 across initialization/training seeds on one fixed split, not cross-validation.

## Speed and saved artifacts

- Record all train/validation/test losses and accuracies every epoch.
- Save `best.pt` immediately whenever validation accuracy strictly improves.
- Save `last.pt` and redraw curves every 25 epochs and at the final epoch, reducing Drive and plotting overhead. Incomplete runs are not resumable.
- Keep deterministic cuDNN settings and FP32 arithmetic; no AMP or altered data policy.
- Save complete configs, environment/source/data hashes, imputation/split metadata, logs, and best/last checkpoints.
- Reload selected checkpoints to produce evaluation-mode train/validation/test scores and confusion matrices.
- Verify every full run's checkpoint predictions, metrics, hashes, constraints, and all confusion counts before accepting it into the comparison table.

The ZIP includes all five runs, frozen selection, individual comparison rows, three-seed means/sample SD, selected-model curves for seeds 42–44, train/validation/test confusion matrices, loss comparison, and EEGNet/DeepConvNet comparison figures. A GPU completion marker means independent local review is still required; it does not establish a performance improvement or award of bonus points.

## Local commands

```bash
python -m unittest discover -s tests -v
python scripts/run_bonus.py --smoke --device cpu --output-dir outputs/verification/bonus_cpu
```

After the full returned ZIP is extracted to a local directory:

```bash
python scripts/bonus_results.py outputs/bonus_review --verify --device cpu
```

The review preserves original GPU verification, reproduces predictions and max-norm constraints, checks configurations and validation selection, and recomputes seed statistics. Original EEGNet figures/results remain separate and unchanged.

## Preparation verification

All 29 regression tests passed locally. All three supplied-data one-epoch CPU smoke runs passed checkpoint reload, source/data hash, preprocessing, max-norm, and train/validation/test confusion checks. The frozen selection/five-run orchestration was tested with simulated full-run scores. Notebook syntax, required ZIP contents, and existing EEGNet source hashes were checked. These are correctness checks, not full GPU accuracy results.

## Verified Public Results

All five returned GPU runs passed independent CPU checkpoint inference, source/data hash checks, planned configuration and fixed-split checks, validation-selection reconstruction, max-norm checks, all confusion counts, and recomputed CSV/JSON statistics. Public snapshots and figures are in `results/bonus/`; original weights, logs, and ZIP remain in `local_archive/` and are excluded from Git.

Rebuild figures from the public snapshots:

```bash
python scripts/bonus_results.py results/bonus
```

Independently verify the private original runs and export portable public snapshots:

```bash
python scripts/bonus_results.py local_archive/training_artifacts/bonus --verify --device cpu --export results/bonus
```

The export omits checkpoints and absolute machine paths while preserving per-run metrics, summary, configuration, environment, source/data hashes, confusion counts, and original GPU/independent CPU verification provenance.
