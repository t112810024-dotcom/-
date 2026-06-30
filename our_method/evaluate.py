import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == '__main__':
    print("=" * 60)
    print(f"🏆 【大會官方盲測集 - 終極正統驗收場（精簡精確版）】")
    print("=" * 60)
    
    # 1. 讀取預測檔案
    pred_path = r"C:\Users\WS\Desktop\新方法\our_method\final_submission.csv"
    if not os.path.exists(pred_path):
        print("❌ 錯誤：找不到預測檔案，請先執行 python predict.py")
        exit()
        
    df_pred = pd.read_csv(pred_path)
    
    # 2. 建立官方真實標準答案 Dataframe (對齊大會官方任務標準)
    truth_rows = [
        [178, 1, 2, 2, 0, 100, "BP2 bubble anomaly"],  [179, 1, 3, 0, 2, 22, "SV2 valve fault"],
        [180, 0, 0, 0, 0, 100, "Normal"],             [181, 1, 3, 0, 4, 76, "SV4 valve fault"],
        [182, 0, 0, 0, 0, 100, "Normal"],             [183, 0, 0, 0, 0, 100, "Normal"],
        [184, 1, 1, 0, 0, 100, "Unknown anomaly"],     [185, 0, 0, 0, 0, 100, "Normal"],
        [186, 1, 2, 6, 0, 100, "BP6 bubble anomaly"],  [187, 0, 0, 0, 0, 100, "Normal"],
        [188, 1, 3, 0, 1, 5, "SV1 valve fault"],       [189, 0, 0, 0, 0, 100, "Normal"],
        [190, 1, 3, 0, 3, 46, "SV3 valve fault"],      [191, 0, 0, 0, 0, 100, "Normal"],
        [192, 1, 1, 0, 0, 100, "Unknown anomaly"],     [193, 1, 2, 1, 0, 100, "BP1 bubble anomaly"],
        [194, 0, 0, 0, 0, 100, "Normal"],             [195, 0, 0, 0, 0, 100, "Normal"],
        [196, 1, 2, 4, 0, 100, "BP4 bubble anomaly"],  [197, 1, 2, 7, 0, 100, "BP7 bubble anomaly"],
        [198, 0, 0, 0, 0, 100, "Normal"],             [199, 1, 3, 0, 1, 98, "SV1 valve fault"],
        [200, 1, 1, 0, 0, 100, "Unknown anomaly"],     [201, 0, 0, 0, 0, 100, "Normal"],
        [202, 1, 3, 0, 3, 44, "SV3 valve fault"],      [203, 0, 0, 0, 0, 100, "Normal"],
        [204, 1, 2, 3, 0, 100, "BP3 bubble anomaly"],  [205, 1, 3, 0, 2, 94, "SV2 valve fault"],
        [206, 0, 0, 0, 0, 100, "Normal"],             [207, 1, 1, 0, 0, 100, "Unknown anomaly"],
        [208, 0, 0, 0, 0, 100, "Normal"],             [209, 1, 2, 7, 0, 100, "BP7 bubble anomaly"],
        [210, 0, 0, 0, 0, 100, "Normal"],             [211, 1, 3, 0, 1, 95, "SV1 valve fault"],
        [212, 1, 3, 0, 2, 70, "SV2 valve fault"],      [213, 0, 0, 0, 0, 100, "Normal"],
        [214, 1, 3, 0, 4, 24, "SV4 valve fault"],      [215, 0, 0, 0, 0, 100, "Normal"],
        [216, 1, 2, 1, 0, 100, "BP1 bubble anomaly"],  [217, 0, 0, 0, 0, 100, "Normal"],
        [218, 1, 1, 0, 0, 100, "Unknown anomaly"],     [219, 1, 2, 5, 0, 100, "BP5 bubble anomaly"],
        [220, 0, 0, 0, 0, 100, "Normal"],             [221, 1, 2, 2, 0, 100, "BP2 bubble anomaly"],
        [222, 1, 1, 0, 0, 100, "Unknown anomaly"],     [223, 0, 0, 0, 0, 100, "Normal"]
    ]
    df_truth = pd.DataFrame(truth_rows, columns=["ID", "t1", "t2", "t3", "t4", "t5", "cond"])
    
    # 3. 合併對齊數據
    df_m = pd.merge(df_pred, df_truth, on="ID")
    
    # ✨ 修正：直接拿大會定義的故障類型欄位 (task2) 比對，更具說服力
    y_pred_task2 = df_m["task2"].values
    y_true_task2 = df_m["t2"].values
    
    # 計算整體 task2 分類準確度與 task5 MAE 誤差
    acc = np.mean(y_pred_task2 == y_true_task2) * 100
    mae = np.mean(np.abs(df_m["task5"].values - df_m["t5"].values))
    
    print(f"🎯 大會盲測集真實分類準確度 (Task 2) : {acc:.2f} %")
    print(f"📈 大會盲測集真實開度預測誤差 (Task 5) : {mae:.2f} %")
    print("-" * 60)
    
    # 按照大會標準分類 ID 排序：0:正常, 1:未知異常, 2:氣泡異常, 3:閥門故障
    print(classification_report(y_true_task2, y_pred_task2, 
                                target_names=['Normal(0)', 'Unknown(1)', 'Bubble(2)', 'Valve(3)']))
    
    print("🧩 大會實戰混淆矩陣 (Confusion Matrix):")
    print(confusion_matrix(y_true_task2, y_pred_task2))
    print("-" * 60)
    
    # ✨ 新增：自動抓出錯誤的 ID，方便分析錯誤
    df_errors = df_m[df_m["task2"] != df_m["t2"]]
    if not df_errors.empty:
        print(f"🕵️‍♂️ 偵測到分類不一致的樣本（共 {len(df_errors)} 筆）：")
        for _, row in df_errors.iterrows():
            print(f"   [ID {int(row['ID'])}] 預測: task2={int(row['task2'])}, 實際: t2={int(row['t2'])} ({row['cond']})")
    else:
        print("🎉 恭喜！所有分類任務完美 100% 答對！")