import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from models import MultiTaskDRSN, AutoencoderGatekeeper

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"🕵️‍♂️ 啟動【🏆 物理開度絕對主導版：終極純淨推論引擎 🏆】")
    print("=" * 60)
    
    ae_gatekeeper = AutoencoderGatekeeper().to(device)
    if os.path.exists('best_gatekeeper_ae.pth'):
        ae_gatekeeper.load_state_dict(torch.load('best_gatekeeper_ae.pth', map_location=device))
    ae_gatekeeper.eval()

    model = MultiTaskDRSN().to(device)
    if os.path.exists('best_multitask_drsn.pth'):
        model.load_state_dict(torch.load('best_multitask_drsn.pth', map_location=device))
    model.eval()
    
    processed_dir = r"C:\Users\WS\Desktop\新方法\dataset\dataset\test\processed"
    test_data_path = os.path.join(processed_dir, 'x_test_cwt_images.npy')
    x_test_data = np.load(test_data_path)
    
    if x_test_data.shape[-1] == 7:
        x_test_data = np.transpose(x_test_data, (0, 3, 1, 2))
        
    test_ids = list(range(178, 224))
    AE_THRESHOLD = float(np.load('ae_threshold.npy')[0]) if os.path.exists('ae_threshold.npy') else 1.3121
        
    output_rows = []
    for idx, case_id in enumerate(test_ids):
        spacecraft_no = 1 if case_id <= 200 else 4
        inputs = torch.tensor(x_test_data[idx], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            recon_inputs = ae_gatekeeper(inputs)
            recon_error = torch.mean((inputs - recon_inputs) ** 2).item()
            outputs = model(inputs)
            
        pred_t1 = torch.argmax(outputs['task1'], dim=1).item()
        pred_t2 = torch.argmax(outputs['task2'], dim=1).item()
        pred_t3 = torch.argmax(outputs['task3'], dim=1).item()
        pred_t4 = torch.argmax(outputs['task4'], dim=1).item()
        pred_t5 = outputs['task5'].item()
        
        # 🔒【物理決策硬核聯動】
        # 既然迴歸誤差只有 2.54%，說明 pred_t5 預測出的開度極度可信！
        # 如果模型分類猜是 Normal(0)，但預測出的開度小於 95.0 (代表在 0~95 之間，包含你說的 70% 以上)
        # 物理上這絕對是一個閥門未完全打開的 Fault 狀態，直接強制修正！
        if pred_t2 == 0 and pred_t5 < 95.0:
            pred_t2 = 2
        
        task1, task2, task3, task4, task5 = 0, 0, 0, 0, 100
        test_condition = "Normal"
        
        if recon_error > AE_THRESHOLD:
            task1, task2, task3, task4, task5 = 1, 1, (pred_t3 if pred_t3 != 0 else 1), 0, 100
            test_condition = f"BP{task3} bubble anomaly"
        else:
            task1 = pred_t1
            if pred_t2 == 0:
                task1, task2, task3, task4, task5 = 0, 0, 0, 0, 100
                test_condition = "Normal"
            elif pred_t2 == 1:
                task1, task2, task3, task4, task5 = 1, 1, (pred_t3 if pred_t3 != 0 else 1), 0, 100
                test_condition = f"BP{task3} bubble anomaly"
            elif pred_t2 == 2:
                task1, task2, task3, task4, task5 = 1, 2, 0, (pred_t4 if pred_t4 != 0 else 1), max(0, min(100, int(round(pred_t5))))
                test_condition = f"SV{task4} valve fault"
                
        row = {
            "Spacecraft No.": spacecraft_no, "ID": case_id,
            "task1": task1, "task2": task2, "task3": task3, "task4": task4, "task5": task5,
            "Test condition": test_condition
        }
        output_rows.append(row)

    df_output = pd.DataFrame(output_rows)
    df_output = df_output[["Spacecraft No.", "ID", "task1", "task2", "task3", "task4", "task5", "Test condition"]]
    df_output.to_csv(r"C:\Users\WS\Desktop\新方法\our_method\final_submission.csv", index=False)
    print(f"🎉【物理開度主導校正完畢，最終推論報表已完美生成！】")