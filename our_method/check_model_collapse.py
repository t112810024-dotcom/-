import os
import torch
import numpy as np
from models import MultiTaskDRSN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processed_dir = r"C:\Users\WS\Desktop\新方法\dataset\dataset\train\processed"
x_train = np.load(os.path.join(processed_dir, 'x_cwt_images.npy'))
y_t2 = np.load(os.path.join(processed_dir, 'y_t2.npy'))

if x_train.shape[-1] == 7:
    x_train = np.transpose(x_train, (0, 3, 1, 2))

model = MultiTaskDRSN().to(device)
model.load_state_dict(torch.load('best_multitask_drsn.pth', map_location=device))
model.eval()

preds = []
with torch.no_grad():
    for i in range(len(x_train)):
        inp = torch.tensor(x_train[i], dtype=torch.float32).unsqueeze(0).to(device)
        out = model(inp)
        pred = torch.argmax(out['task2'], dim=1).item()
        preds.append(pred)

preds = np.array(preds)
print("模型在訓練集上的 t2 預測分布:", np.bincount(preds, minlength=3))
print("訓練集真實 t2 分布:          ", np.bincount(y_t2, minlength=3))

# 看每個真實類別,模型分別預測成什麼
for true_cls in range(3):
    mask = (y_t2 == true_cls)
    pred_for_this = preds[mask]
    print(f"真實類別={true_cls} (共{mask.sum()}筆) → 模型預測分布: {np.bincount(pred_for_this, minlength=3)}")