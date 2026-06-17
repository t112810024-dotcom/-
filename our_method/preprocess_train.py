import os
import numpy as np
import pandas as pd
import pywt

if __name__ == '__main__':
    label_path = r"C:\Users\WS\Desktop\新方法\dataset\dataset\train\labels.xlsx"
    data_dir = r"C:\Users\WS\Desktop\新方法\dataset\dataset\train\data"
    output_dir = r"C:\Users\WS\Desktop\新方法\dataset\dataset\train\processed"

    os.makedirs(output_dir, exist_ok=True)

    print("🚀 [五任務特徵對齊] 開始解析合法訓練集標籤...")
    column_names = ['Case', 'Spacecraft', 'Condition', 'SV1', 'SV2', 'SV3', 'SV4',
                    'BP1', 'BP2', 'BP3', 'BP4', 'BP5', 'BP6', 'BP7', 'BV1']
    df_labels = pd.read_excel(label_path, header=None, skiprows=2, names=column_names)
    df_labels = df_labels.dropna(subset=['Case'])
    df_labels['Case'] = df_labels['Case'].astype(int)

    num_channels = 7
    total_scales = 64
    wavelet_name = 'morl'
    scales = np.arange(1, total_scales + 1)

    features = []
    t1_list, t2_list, t3_list, t4_list, t5_list = [], [], [], [], []

    print("⏳ 正在執行通道級張量標準化並編碼『五任務整合標籤』...")
    for idx, row in df_labels.iterrows():
        case_num = int(row['Case'])
        file_name = f"Case{case_num:03d}.csv"
        file_path = os.path.join(data_dir, file_name)

        if not os.path.exists(file_path):
            print(f"⚠️ 找不到 {file_name}，跳過")
            continue

        df_data = pd.read_csv(file_path)
        pressure_matrix = df_data[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']].values.T

        eps = 1e-8
        mean = np.mean(pressure_matrix, axis=1, keepdims=True)
        std = np.std(pressure_matrix, axis=1, keepdims=True)
        normalized_matrix = (pressure_matrix - mean) / (std + eps)

        cwt_feat = np.zeros((num_channels, total_scales, 1201), dtype=np.float32)
        for ch in range(num_channels):
            coef, _ = pywt.cwt(normalized_matrix[ch], scales, wavelet_name)
            cwt_feat[ch, :, :] = np.abs(coef)

        features.append(cwt_feat)

        # ✅ 修正：根據實際欄位格式解析
        cond_str = str(row['Condition']).strip()

        task1 = 0
        task2 = 0
        task3 = 0
        task4 = 0
        task5 = 100.0

        if cond_str == 'Normal':
            # 全部保持預設值
            pass

        elif cond_str == 'Anomaly':
            # ✅ 氣泡異常：BP欄位是 'Yes'/'No'
            task1 = 1
            task2 = 1  # 1=氣泡
            for bp_idx in range(1, 8):
                if str(row[f'BP{bp_idx}']).strip() == 'Yes':
                    task3 = bp_idx
                    break

        elif cond_str == 'Fault':
            # ✅ 閥門故障：SV欄位是數值，找開度小於100的
            task1 = 1
            task2 = 2  # 2=閥門
            for sv_idx in range(1, 5):
                val = float(row[f'SV{sv_idx}'])
                if val < 100.0:
                    task4 = sv_idx
                    task5 = val
                    break

        t1_list.append(task1)
        t2_list.append(task2)
        t3_list.append(task3)
        t4_list.append(task4)
        t5_list.append(task5)

    arr_features = np.array(features)
    arr_t1 = np.array(t1_list, dtype=np.int64)
    arr_t2 = np.array(t2_list, dtype=np.int64)
    arr_t3 = np.array(t3_list, dtype=np.int64)
    arr_t4 = np.array(t4_list, dtype=np.int64)
    arr_t5 = np.array(t5_list, dtype=np.float32)

    arr_weak_mask = (arr_t2 == 2) & (arr_t5 >= 70.0) & (arr_t5 < 100.0)

    np.save(os.path.join(output_dir, 'x_cwt_images.npy'), arr_features)
    np.save(os.path.join(output_dir, 'y_t1.npy'), arr_t1)
    np.save(os.path.join(output_dir, 'y_t2.npy'), arr_t2)
    np.save(os.path.join(output_dir, 'y_t3.npy'), arr_t3)
    np.save(os.path.join(output_dir, 'y_t4.npy'), arr_t4)
    np.save(os.path.join(output_dir, 'y_t5.npy'), arr_t5)
    np.save(os.path.join(output_dir, 'y_weak_mask.npy'), arr_weak_mask)

    print("\n" + "=" * 60)
    print(f"🎉【完成！】檔案已寫入：\n💾 {output_dir}")
    print(f"📊 總計處理：{len(features)} 筆訓練樣本")
    print(f"  - Normal  : {np.sum(arr_t2 == 0)} 筆")
    print(f"  - Anomaly : {np.sum(arr_t2 == 1)} 筆")
    print(f"  - Fault   : {np.sum(arr_t2 == 2)} 筆")
    print(f"  - t3 分布 : {np.bincount(arr_t3)}")
    print(f"  - t4 分布 : {np.bincount(arr_t4)}")
    print(f"🎯 偵測到微弱邊緣故障樣本數：{np.sum(arr_weak_mask)} 筆")
    print("=" * 60)