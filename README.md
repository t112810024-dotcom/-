
# 新方法 (Our Method)

此 README 為詳細說明，包含專案概述、資料格式、每支腳本的用途與執行步驟（含範例指令），以及常見問題與建議設定。

## 1. 專案概述

本專案提供一套用於時序壓力感測資料的前處理、模型訓練、多任務推論（分類 + 回歸）、以及 Grad-CAM 可視化的完整流程。主要程式位於 `our_method`，資料位於 `dataset`。

主要流程：
- 先用 `our_method/preprocess.py` 由原始 CSV 生成 CWT 特徵張量（儲存為 .npy）
- 使用 `our_method/train.py` 訓練模型（產出 `best_multitask_drsn.pth`）
- 使用 `our_method/predict.py` 對測試集進行推論並輸出 Excel（同時產生 Grad-CAM 圖）
- 使用 `our_method/evaluate.py` 與提供的 ground-truth (`dataset/answer/answer.csv`) 比較，計算 accuracy/MAE 等指標

## 2. 環境與相依套件

建議建立乾淨的 Python 虛擬環境（Windows 範例）：

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -U pip
```

推薦套件（可放入 `requirements.txt`）：

```
numpy
pandas
pywt
torch
torchvision
scikit-learn
matplotlib
opencv-python
openpyxl
```

備註：`torch` 的版本請依 GPU 與 CUDA 版本選擇相容的 wheel，或用 CPU 版本。

## 3. 資料結構與格式

工作目錄下重要路徑：
- `dataset/train/data/` : 訓練用 CSV（每個 Case 一個檔案，檔名例如 Case001.csv 或 Case1.csv）
- `dataset/train/labels.xlsx` : 訓練標籤檔案（`preprocess.py` 期望以特定欄位順序讀取，見下）
- `dataset/train/processed/` : `preprocess.py` 會在此輸出三個檔案：
  - `x_cwt_images.npy` : numpy array, shape [N, 7, 128, 1201]
  - `y_labels_cls.npy`  : 類別標籤陣列（文字，如 'Normal'/'Fault'/'Anomaly'）
  - `y_labels_reg.npy`  : 回歸目標（SV1）
- `dataset/test/data/`  : 測試集 CSV（predict.py 會掃描 Case 範圍）
- `dataset/answer/answer.csv` : 測試集真實標籤（evaluate.py 期望含 `ID` 與 `task5` 欄位）

CSV 檔案格式（每個 Case 的 CSV）：應包含欄位 `P1`..`P7` 為感測通道時間序列，程式會取這 7 個欄位並做 CWT。

labels.xlsx 的前幾欄預期順序：`Case, Spacecraft, Condition, SV1, SV2, ...`（`preprocess.py` 以此對應，並從第三列開始讀取）。

## 4. 腳本使用說明（詳細）

- `our_method/preprocess.py`
  - 作用：讀取 `dataset/train/data/` 與 `dataset/train/labels.xlsx`，對每個 Case 計算 CWT（continuous wavelet transform），並輸出三個 .npy 檔供訓練使用。
  - 輸入：`data_dir` 與 `label_path` 在檔案頭被設定為絕對路徑，可直接修改變數或改寫成 argparse。
  - 輸出：`dataset/train/processed/x_cwt_images.npy`, `y_labels_cls.npy`, `y_labels_reg.npy`

- `our_method/train.py`
  - 作用：載入 `dataset/train/processed` 的 .npy 檔，建立 `AutoencoderGatekeeper` 與 `MultiTaskDRSN`，先訓練 AE，再訓練 DRSN，最後儲存 `best_multitask_drsn.pth`。
  - 重要設定（檔頭）：`BATCH_SIZE`, `AE_EPOCHS`, `DRSN_EPOCHS`, `LEARNING_RATE`。可直接在檔案內調整或改為 CLI 參數。
  - 執行範例：

```powershell
python our_method/train.py
```

  - 輸出：`best_multitask_drsn.pth`（存於執行目錄）

- `our_method/predict.py`
  - 作用：載入訓練好的模型 `best_multitask_drsn.pth`，對測試資料做 CWT 預處理、推論分類與回歸，輸出 Excel 檔 `dataset/test/predict_results.xlsx`，並在 `our_method/gradcam_results/` 儲存 Grad-CAM 可視化圖。
  - 預設會掃描範圍 `Case178` 至 `Case223`（可在程式內調整）並嘗試以 `Case{num}.csv` 或 `Case{num:03d}.csv` 找檔案。
  - 執行範例：

```powershell
python our_method/predict.py
```

  - 輸出：`dataset/test/predict_results.xlsx`（含 `Case`, `Predicted_Condition`, `Predicted_SV1_Opening(%)`）

- `our_method/evaluate.py`
  - 作用：讀取 `predict_results.xlsx` 與 `dataset/answer/answer.csv`，合併後計算分類的 accuracy、classification report、混淆矩陣，以及回歸的 MAE（對 `SV1/task5`）。
  - 執行範例：

```powershell
python our_method/evaluate.py
```

  - 主要輸出：終端列印 Final Test Accuracy 與 Final Test MAE，以及分類報表與混淆矩陣。

## 5. 範例完整流程（從頭到尾）

1. 建立環境並安裝套件（見第 2 節）
2. 產生訓練特徵：

```powershell
python our_method/preprocess.py
```

3. 訓練模型（注意需有足夠記憶體與 GPU 建議）：

```powershell
python our_method/train.py
```

4. 使用訓練好的模型對測試集推論：

```powershell
python our_method/predict.py
```

5. 評估結果：

```powershell
python our_method/evaluate.py
```

## 6. 常見問題與排查建議

- 若 `preprocess.py` 找不到 `labels.xlsx`：請確保檔案路徑正確，且 Excel 的標頭與程式預期一致（程式會跳過前兩列）。
- 若訓練過程 GPU 記憶體不足：
  - 減少 `BATCH_SIZE`（在 `train.py` 中修改）或在 CPU 上執行（速度會慢）。
- 若 `predict.py` 沒有輸出某些 Case：程式會跳過缺失的 CSV 檔，請確認 `dataset/test/data/` 中檔名是否為 `CaseNNN.csv` 或 `CaseN.csv`。

## 7. 模型與輸出檔案

- `our_method/best_multitask_drsn.pth`：多任務模型權重（分類 + 回歸）
- `our_method/best_gatekeeper_ae.pth`：自編碼器權重（若已儲存）
- `dataset/test/predict_results.xlsx`：推論結果
- `our_method/gradcam_results/`：Grad-CAM 圖像

## 8. 想要我幫你做的事？

- 我可以：
  - 幫你根據目前程式自動生成 `requirements.txt`（我會根據 import 偵測套件）
  - 將各腳本改寫為支援 CLI 參數（argparse），並更新 README 的對應範例
  - 將 README 翻譯成英文或新增英文版 `README_en.md`

請回覆你希望的下一步，我會繼續執行。

