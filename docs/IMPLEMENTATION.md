# 模型與資料處理

## EEGNet

程式位置：[`models/EEGNet.py`](../lab2_EEG_classification/models/EEGNet.py)。輸入為 `(B, 1, 2, 750)`，輸出為兩類 raw logits，配合 cross-entropy loss，不在模型尾端加 softmax。

| 區塊 | 實作 | 輸出形狀（不含 batch） |
| --- | --- | --- |
| Temporal convolution | Conv2d 1→16，kernel `(1,51)`，padding `(0,25)`；BatchNorm | `(16,2,750)` |
| Spatial depthwise convolution | Conv2d 16→32，kernel `(2,1)`，groups=16；BatchNorm、activation、AvgPool `(1,4)`、dropout | `(32,1,187)` |
| 第二個 temporal block | 預設 Conv2d 32→32，kernel `(1,15)`，padding `(0,7)`；BatchNorm、activation、AvgPool `(1,8)`、dropout | `(32,1,23)` |
| Classifier | Flatten 736 → Linear 2 | `(2,)` |

預設共 **17,874** 個可訓練參數。Spatial convolution 的 groups=16 讓每個 temporal feature map 分別學習空間濾波，depth multiplier 為 2。

課程投影片雖將第二個 temporal block 命名為 `separableConv`，圖示使用的卻是一般 convolution（groups=1）。因此預設 `temporal_block="slide"` 忠實實作投影片。另提供 `"depthwise_separable"`，用 32-group temporal convolution 加 1×1 pointwise convolution，參數降至 **4,018**。兩者在實驗中分開比較。

## Activation

- ELU：正值區域維持線性，負值區域使用 `alpha * (exp(x) - 1)`；比較 alpha 0.5、1、2。
- ReLU：`max(0, x)`。
- LeakyReLU：正值保持線性，負值保留 slope 0.01，避免負值區域的梯度全部為零。

Activation 設定同時套用到兩個有 activation 的 convolution blocks。實驗結論以本資料、架構與設定下的結果為限。

## 資料流程

[`dataloader.py`](../lab2_EEG_classification/dataloader.py) 載入兩位受試者（S4b、X11b）的 supplied train/test。每個 NPZ 有 540 trials；合併後 supplied train 和 test 各 **1,080** 筆。

1. 標籤 1/2 轉成 0/1，訊號轉成 `(N,1,2,750)`。
2. 先以 subject＋class 分層，從 supplied train 切出 **864 train／216 validation**。
3. split seed 固定為 42，與控制初始化／訓練的 model seed 分開。
4. 只用 train partition 計算 scalar mean，填補 train、validation、test 的 NaN。
5. 不另外 normalization。Test 保持課程提供的 1,080 筆。

## 訓練與評估

[`main.py`](../lab2_EEG_classification/main.py) 使用 Adam、cross-entropy、150 epochs，儲存每個 epoch 的 Loss／Accuracy、完整設定、source/data hashes、checkpoint 與曲線。Loss 依樣本數加權平均。

- `best.pt`：Validation Accuracy 最高的 checkpoint；相同 accuracy 時保留最早 epoch。
- `last.pt`：最後一個 epoch。
- 設定選擇使用 validation，test 不參與選模。
- `metrics.csv` 的 Train Loss／Accuracy 是訓練過程（dropout 開啟、參數持續更新）量測值。
- `summary.json` 的 selected-checkpoint scores 對三個分割全部使用 evaluation mode。這與訓練曲線中的 Train 指標有不同量測條件。
- 每個 epoch 的 test 曲線與最大值為作業分析資料。最高 monitored test accuracy 是多次評估的最大值，不能當成新的獨立 held-out estimate。

評估 DataLoader 使用獨立 RNG，避免評估改變後續 dropout 的隨機序列。完整訓練程式保留原始已驗證版本；整理目錄時未更動其內容。
