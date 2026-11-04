# EEGNet：腦波訊號分類

自行實作 EEGNet，使用課程提供的 BCI Competition III、IIIb EEG 資料，比較 activation、dropout、learning rate 與卷積架構，並以三個 model seeds 檢查結果一致性。

## 主要結果

以下為固定資料切分、model seeds **42、43、44** 的平均值 ± 樣本標準差；checkpoint 一律由 Validation Accuracy 選出。

| 設定 | Validation Accuracy | Test Accuracy |
| --- | ---: | ---: |
| ELU baseline | 82.41% ± 1.22% | 77.47% ± 1.54% |
| **LeakyReLU（選定設定）** | **88.27% ± 0.53%** | **84.51% ± 0.51%** |

LeakyReLU 在三個 paired seeds 的 validation 與 test accuracy 均優於 baseline，平均 test accuracy 提高 **7.04 個百分點**。三個 model seeds 共用同一個 split，因此不是 cross-validation；seed 42 曾用於調參，也納入平均值。

選定設定：LeakyReLU slope **0.01**、dropout **0.25**、Adam learning rate **0.01**、weight decay **0.001**、batch size **64**、**150 epochs**，共 **17,874** 個參數。

- **選定模型範例：**seed 42、validation-selected epoch 131；Train **98.73%**、Validation **88.89%**、Test **85.00%**。
- **最高 monitored test accuracy：86.11%**（seed 44、epoch 123）。這是多次 test 評估中的最大值，與 validation-selected checkpoint 的成績、三個 seeds 平均值分開呈現。

![三個 seeds 的比較](results/figures/seed_comparison.png)

## 閱讀入口

- [完整實驗結果與討論](results/README.md)：17 組完整實驗、組合比較、三個 seeds、Loss 與 confusion matrix。
- [模型與資料處理](docs/IMPLEMENTATION.md)：EEGNet 架構、activation、depthwise／separable convolution、評估方式。
- [如何重現](docs/REPRODUCIBILITY.md)：資料放置、訓練、Colab、checkpoint 驗證、圖表重建。
- [選定模型完整曲線](results/figures/selected_curves.png)｜[Confusion matrix](results/figures/confusion_matrices.png)｜[全部 Loss 曲線](results/figures/all_loss_curves.png)。

## 專案結構

```text
lab2_EEG_classification/   EEGNet、資料處理與訓練主程式
configs/                  baseline、選定模型、單因子與組合實驗設定
notebooks/Colab.ipynb      單一 Colab 工作流程
scripts/                  批次訓練、驗證、結果整理與 Colab 打包
results/                  實驗總表、圖表、每個實驗的逐 epoch 數據
  runs/                   17 個實驗的設定、metrics、summary 與驗證紀錄
  figures/                Loss／Accuracy、seed 比較及 confusion matrix
  README.md               統整後的實驗分析
  experiments.csv         完整實驗總表
  loss_comparison.csv     各實驗的 Loss
  seed_summary.csv        三個 seeds 的平均與標準差
docs/                     實作與重現說明
tests/                    資料、模型、訓練與結果一致性測試
```

資料集、模型權重、原始回傳 ZIP 與本機工作備份不納入 Git。根目錄的 `local_archive/` 是本機封存區，`outputs/` 是新實驗輸出位置，兩者均由 `.gitignore` 排除。已審查的 `results/` 與圖表會納入 Git。

## 快速開始

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

測試使用合成小型資料，不需要先取得完整資料集。要訓練，請把課程提供的四個 NPZ 放到 [`lab2_EEG_classification/data/`](lab2_EEG_classification/data/README.md)，再執行：

```bash
python lab2_EEG_classification/main.py --config configs/selected.json --device cuda
```

批次訓練與三個 seeds：

```bash
python scripts/run_experiments.py --configs configs/baseline.json configs/selected.json --seeds 42 43 44 --device cuda
```

使用 Colab 時，執行 `python scripts/package_colab.py`，再開啟 [`notebooks/Colab.ipynb`](notebooks/Colab.ipynb)。

不用訓練或資料集，也可從已公開的 metrics／summary 重建圖表與總表：

```bash
python scripts/analyze_results.py
```
