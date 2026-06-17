import os
import numpy as np
import pandas as pd
import pywt

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 [真·盲測特徵轉換] 開始處理大會 46 筆未知測試集原始訊號...")
    print("=" * 60)

    # 🔧 1. 請確認你的測試集原始 .csv 檔案（Case178.csv ~ Case223.csv）存放的路徑
    test_data_dir = r"C:\Users\WS\Desktop\新方法\dataset\dataset\test\data"  
    
    # 💡 優化點：依據全新流程圖設計，將測試集轉換後的特徵存放在獨立的 test\processed 檔案夾中
    output_dir = r"C:\Users\WS\Desktop\新方法\dataset\dataset\test\processed"
    os.makedirs(output_dir, exist_ok=True)

    # 🔧 2. CWT 核心配置（🌟 與模型 Backbone 的輸入特徵尺寸 64x1201 完全對齊）
    num_channels = 7        
    total_scales = 64      # 💡 修正點：從 128 修正為 64，完美對齊全新 DRSN 模型的輸入要求
    wavelet_name = 'morl'  
    scales = np.arange(1, total_scales + 1)

    features_test = []
    test_ids = list(range(178, 224)) # Case 178 到 223

    for case_num in test_ids:
        file_name = f"Case{case_num:03d}.csv"
        file_path = os.path.join(test_data_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"❌ 錯誤：找不到盲測集原始檔案 {file_path}，請檢查路徑！")
            continue
            
        print(f"⏳ 正在轉換 {file_name} ...")
        df_data = pd.read_csv(file_path)
        pressure_matrix = df_data[['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7']].values.T
        
        # 💡 核心通道級 z-score 標準化（Instance Normalization）
        eps = 1e-8
        mean = np.mean(pressure_matrix, axis=1, keepdims=True)
        std = np.std(pressure_matrix, axis=1, keepdims=True)
        normalized_matrix = (pressure_matrix - mean) / (std + eps)
        
        # 💡 建立 (7, 64, 1201) 空間的特徵圖
        cwt_feat = np.zeros((num_channels, total_scales, 1201), dtype=np.float32)
        for ch in range(num_channels):
            coef, _ = pywt.cwt(normalized_matrix[ch], scales, wavelet_name)
            cwt_feat[ch, :, :] = np.abs(coef)
            
        features_test.append(cwt_feat)

    if len(features_test) == 46:
        # 儲存為獨立的測試集矩陣
        output_path = os.path.join(output_dir, 'x_test_cwt_images.npy')
        np.save(output_path, np.array(features_test))
        print("\n" + "=" * 60)
        print(f"🎉【盲測特徵轉換成功！】46 筆特徵已安全寫入：\n💾 {output_path}")
        print("=" * 60)
    else:
        print(f"\n⚠️ 轉換未完全！預期 46 筆，實際成功 {len(features_test)} 筆。請檢查數據夾路徑。")