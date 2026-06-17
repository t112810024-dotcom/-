import os
import torch
import numpy as np
from models import MultiTaskDRSN

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processed_dir = r"C:\Users\WS\Desktop\新方法\dataset\dataset\train\processed"

    x_train = np.load(os.path.join(processed_dir, 'x_cwt_images.npy'))
    y_t2 = np.load(os.path.join(processed_dir, 'y_t2.npy'))
    y_t5 = np.load(os.path.join(processed_dir, 'y_t5.npy'))

    if x_train.shape[-1] == 7:    
        x_train = np.transpose(x_train, (0, 3, 1, 2))

    model = MultiTaskDRSN().to(device)
    model.load_state_dict(torch.load('best_multitask_drsn.pth', map_location=device))
    model.eval()

    preds = []
    probs_list = []

    with torch.no_grad():    
        for i in range(len(x_train)):        
            inp = torch.tensor(x_train[i], dtype=torch.float32).unsqueeze(0).to(device)        
            out = model(inp)        
            prob = torch.softmax(out['task2'], dim=1).cpu().numpy()[0]        
            pred = np.argmax(prob)        
            preds.append(pred)        
            probs_list.append(prob)

    preds = np.array(preds)
    probs_arr = np.array(probs_list)

    # 找出真實是 Normal(0) 但被誤判成 Fault(2) 的樣本
    mis_normal = np.where((y_t2 == 0) & (preds == 2))[0]
    print("=" * 60)
    print(f"📊 共有 {len(mis_normal)} 筆 Normal 被誤判成 Fault")
    print("這些樣本的模型機率分布 [Normal, Anomaly, Fault]:")
    for idx in mis_normal[:10]:    
        print(f"  idx={idx}: probs={probs_arr[idx].round(3)}")

    # 找出真實是 Fault(2) 且 t5(開度) 較高的樣本，看模型表現
    high_opening_fault = np.where((y_t2 == 2) & (y_t5 >= 70))[0]
    print("-" * 60)
    print(f"📊 共有 {len(high_opening_fault)} 筆高開度Fault樣本 (t5>=70)")
    for idx in high_opening_fault:    
        print(f"  idx={idx}: t5={y_t5[idx]:.1f}, pred={preds[idx]}, probs={probs_arr[idx].round(3)}")
    print("=" * 60)