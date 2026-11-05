# 重現與驗證

## 安裝與資料

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

本機測試過的訓練流程使用 Python/PyTorch；各完整實驗實際套件、GPU 與 source/data hashes 記錄在 `results/runs/*/environment.json`。requirements 是相容依賴範圍，不是鎖定環境；不同硬體／版本可能產生數值差異。

將課程提供的 `S4b_train.npz`、`X11b_train.npz`、`S4b_test.npz`、`X11b_test.npz` 放進 `lab2_EEG_classification/data/`。GitHub 不包含 dataset 或 checkpoint。

## 測試

```bash
python -m unittest discover -s tests -v
```

模型、資料切分、訓練、checkpoint 與結果一致性測試使用合成小型資料或已公開數據，不依賴完整 dataset／GPU。

## 訓練

```bash
# 短 CPU smoke check；不是完整實驗成績
python lab2_EEG_classification/main.py --config configs/selected.json --epochs 1 --device cpu --num-threads 2

# 三個 model seeds；split seed 仍為 42
python scripts/run_experiments.py --configs configs/baseline.json configs/selected.json --seeds 42 43 44 --device cuda

# 九個單因子比較
python scripts/run_experiments.py --configs configs/experiments/*.json --device cuda

# 三個 LeakyReLU 組合
python scripts/run_experiments.py --configs configs/combinations/*.json --device cuda
```

每次訓練使用獨立 timestamp 目錄。批次 runner 重新執行時，會驗證並沿用同一 output directory 中設定完全相符的完整 run。Incomplete run 從 epoch 1 重跑；不支援中途 checkpoint 續訓。多個相符的完整 run 會要求先處理歧義，而不是隨意挑選。不要同時對同一 output directory 執行多個 runner。

訓練 CLI 支援 `--epochs`、`--batch-size`、`--lr`、`--seed`、`--output-dir`、`--device`、`--num-threads`。設定內 data/output 相對路徑以專案根目錄解析，`--config` 相對路徑則以執行時工作目錄解析。

## Colab

1. 本機放好四個 NPZ，執行 `python scripts/package_colab.py`。
2. 開啟 `notebooks/Colab.ipynb`，選擇 GPU runtime，依序上傳 `outputs/Lab2_colab.zip`、安裝與掛載 Drive。
3. 執行短 GPU verification。
4. 只有要新增完整實驗時才把 `RUN_FULL_EXPERIMENTS` 改成 `True`，並明確指定 configs/seeds。

現有實驗均已完成，不必為閱讀結果重新訓練。私人上傳 ZIP 含 dataset，不應發布；原始 checkpoint 留在 Drive／本機。

## 兩種驗證層級

**已公開結果的數據一致性與圖表重建：**

```bash
python scripts/analyze_results.py
```

不需要 dataset、checkpoint 或 GPU。程式檢查每個 run 的 150-epoch metrics、validation-selected epoch、test maximum、summary 與固定 split，再重新計算 CSV、三個 seeds 的平均／sample SD，重繪所有圖表。Confusion matrix 圖從已驗證的公開計數重繪，不是重新執行模型。

**原始 run 的完整 checkpoint 驗證：**

```bash
python scripts/verify_run.py outputs/experiments/YOUR_RUN_FOLDER --data-dir lab2_EEG_classification/data --device cpu
```

需要原始完整 run（包含 `best.pt`、`last.pt`、`preprocessing.json` 等）和 dataset。會核對 source/data hashes、切分、指標、checkpoint predictions 與圖表；CLI 將本機驗證寫到 `local_review.json`，保留既有的 GPU `verification.json`。公開 `results/runs/` 是精簡審查快照，不能直接取代原始完整 run。

重新執行選定模型的 confusion matrix 推論：

```bash
python scripts/analyze_results.py --checkpoint /path/to/original/seed42/best.pt
```

程式核對原始 checkpoint SHA-256、source/data hashes、imputation／split 與 epoch，確認 Train／Validation／Test 的 Loss、Accuracy 和 confusion counts 都與已保存數據一致。使用 `weights_only=True` 載入權重。

## 公開快照與本機封存

每個 `results/runs/<experiment>/` 包含 `config.json`、`metrics.csv`、`summary.json`、`environment.json`、`verification.json`。公開設定的名稱已統一，data/output paths 改為相對路徑；原始訓練的 experiment name 和 checkpoint hashes 留在 verification provenance，原始指標／summary 未改寫。公開快照不含權重或 EEG 訊號。

`local_archive/` 是整理前資料的本機封存，已被 Git 忽略。原始訓練成果在 `local_archive/training_artifacts/`，`checkpoints.json` 記錄各設定的本機 checkpoint 路徑；歷史文件與舊交接程式可由 `before_cleanup.zip` 還原。課程 PDF、回傳 ZIP、報告草稿放在 `local_archive/source_materials/`，未刪除或修改內容。

## DeepConvNet Bonus

已完成且獨立驗證的五組 DeepConvNet 結果集中於 [`results/bonus/`](../results/bonus/README.md)，與原本 17 組 EEGNet 結果分開保存。

```bash
# 不需 dataset／checkpoint，從公開快照重繪 Bonus 圖表
python scripts/bonus_results.py results/bonus

# 要重跑完整 GPU 實驗時才執行；既有 EEGNet 不會重訓
python scripts/run_bonus.py --output-dir outputs/bonus --device cuda
```

架構、選模規則、Colab 與完整驗證／匯出方式見 [Bonus 文件](BONUS.md)。
