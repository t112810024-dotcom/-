import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models import MultiTaskDRSN, AutoencoderGatekeeper

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("🚀 啟動【五任務 DRSN 多任務學習平衡訓練引擎】")
    print("=" * 60)

    processed_dir = r"C:\Users\WS\Desktop\新方法\dataset\dataset\train\processed"

    x_train = np.load(os.path.join(processed_dir, 'x_cwt_images.npy'))
    y_t1 = np.load(os.path.join(processed_dir, 'y_t1.npy'))
    y_t2 = np.load(os.path.join(processed_dir, 'y_t2.npy'))
    y_t3 = np.load(os.path.join(processed_dir, 'y_t3.npy'))
    y_t4 = np.load(os.path.join(processed_dir, 'y_t4.npy'))
    y_t5 = np.load(os.path.join(processed_dir, 'y_t5.npy'))
    y_weak_mask = np.load(os.path.join(processed_dir, 'y_weak_mask.npy'))

    if x_train.shape[-1] == 7:
        x_train = np.transpose(x_train, (0, 3, 1, 2))

    # ===== 階段一：AE 守門員訓練 =====
    print("⏳ 階段一：訓練 Autoencoder 守門員...")
    normal_indices = (y_t2 == 0)
    x_train_normal = x_train[normal_indices]
    ae_dataset = TensorDataset(torch.tensor(x_train_normal, dtype=torch.float32))
    ae_loader = DataLoader(ae_dataset, batch_size=16, shuffle=True)

    ae_model = AutoencoderGatekeeper().to(device)
    ae_optimizer = optim.AdamW(ae_model.parameters(), lr=0.001, weight_decay=1e-4)
    ae_criterion = nn.MSELoss()

    for epoch in range(30):
        ae_model.train()
        for inputs in ae_loader:
            inputs = inputs[0].to(device)
            ae_optimizer.zero_grad()
            recon = ae_model(inputs)
            loss = ae_criterion(recon, inputs)
            loss.backward()
            ae_optimizer.step()
    torch.save(ae_model.state_dict(), 'best_gatekeeper_ae.pth')
    print("✅ 守門員訓練完畢！")

    # ===== 計算動態閾值 =====
    print("⏳ 計算 AE 動態閾值...")
    ae_model.eval()
    normal_errors = []
    anomaly_errors = []
    with torch.no_grad():
        for i in range(len(x_train)):
            inp = torch.tensor(x_train[i], dtype=torch.float32).unsqueeze(0).to(device)
            recon = ae_model(inp)
            err = torch.mean((inp - recon) ** 2).item()
            if y_t2[i] == 0:
                normal_errors.append(err)
            else:
                anomaly_errors.append(err)

    normal_errors = np.array(normal_errors)
    anomaly_errors = np.array(anomaly_errors)
    ae_threshold = float(np.mean(normal_errors) + 3 * np.std(normal_errors))

    print(f"  Normal  重建誤差: mean={np.mean(normal_errors):.4f}, std={np.std(normal_errors):.4f}")
    print(f"  Anomaly 重建誤差: mean={np.mean(anomaly_errors):.4f}, std={np.std(anomaly_errors):.4f}")
    print(f"  ✅ 自動閾值設定為: {ae_threshold:.4f}")
    np.save('ae_threshold.npy', np.array([ae_threshold]))

    # ===== 階段二：五任務 DRSN =====
    print("⏳ 階段二：訓練五任務 DRSN...")

    t2_counts = np.bincount(y_t2)
    t2_weight = torch.tensor(
        [len(y_t2) / (len(t2_counts) * c) for c in t2_counts],
        dtype=torch.float32
    ).to(device)

    t3_counts = np.bincount(y_t3)
    t3_weight = torch.tensor(
        [len(y_t3) / (len(t3_counts) * max(c, 1)) for c in t3_counts],
        dtype=torch.float32
    ).to(device)

    t4_counts = np.bincount(y_t4)
    t4_weight = torch.tensor(
        [len(y_t4) / (len(t4_counts) * max(c, 1)) for c in t4_counts],
        dtype=torch.float32
    ).to(device)

    print(f"  t2 class weight: {t2_weight.cpu().numpy().round(3)}")
    print(f"  t3 class weight: {t3_weight.cpu().numpy().round(3)}")
    print(f"  t4 class weight: {t4_weight.cpu().numpy().round(3)}")

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_t1, dtype=torch.long),
        torch.tensor(y_t2, dtype=torch.long),
        torch.tensor(y_t3, dtype=torch.long),
        torch.tensor(y_t4, dtype=torch.long),
        torch.tensor(y_t5, dtype=torch.float32),
        torch.tensor(y_weak_mask, dtype=torch.bool)
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = MultiTaskDRSN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

    criterion_t1 = nn.CrossEntropyLoss(reduction='none')
    criterion_t2 = nn.CrossEntropyLoss(weight=t2_weight, reduction='none')
    criterion_t3 = nn.CrossEntropyLoss(weight=t3_weight, reduction='none')
    criterion_t4 = nn.CrossEntropyLoss(weight=t4_weight, reduction='none')
    criterion_mse = nn.HuberLoss(reduction='none', delta=10.0)

    best_t2_acc = 0.0
    epochs = 120

    # 🎯 物理邊界優化點（一）：調降 W_T2 權重至 1.0，防止邊界嚴重傾斜過度防禦
    W_T1, W_T2, W_T3, W_T4, W_T5 = 1.0, 1.0, 1.0, 1.0, 0.05

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct_t2 = 0

        for inputs, t1, t2, t3, t4, t5, weak_mask in train_loader:
            inputs = inputs.to(device)
            t1, t2, t3, t4, t5 = t1.to(device), t2.to(device), t3.to(device), t4.to(device), t5.to(device)
            weak_mask = weak_mask.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss_t1 = criterion_t1(outputs['task1'], t1).mean()
            loss_t2 = criterion_t2(outputs['task2'], t2).mean()
            loss_t3 = criterion_t3(outputs['task3'], t3).mean()
            loss_t4 = criterion_t4(outputs['task4'], t4).mean()
            loss_t5 = criterion_mse(outputs['task5'], t5).mean()

            # 🎯 物理邊界優化點（二）：調降微弱故障加權倍率由 8.0 改為平衡的 3.0，拉回決策分佈
            if torch.any(weak_mask):
                loss_t2_raw = criterion_t2(outputs['task2'], t2)
                loss_t5_raw = criterion_mse(outputs['task5'], t5)
                loss_t2 = torch.where(weak_mask, loss_t2_raw * 3.0, loss_t2_raw).mean()
                loss_t5 = torch.where(weak_mask, loss_t5_raw * 3.0, loss_t5_raw).mean()

            total_loss = (W_T1 * loss_t1 + W_T2 * loss_t2 + W_T3 * loss_t3 +
                          W_T4 * loss_t4 + W_T5 * loss_t5)

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

            pred_t2 = torch.argmax(outputs['task2'], dim=1)
            correct_t2 += (pred_t2 == t2).sum().item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_acc_t2 = correct_t2 / len(y_t2) * 100

        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] | 損耗: {avg_loss:.4f} | t2訓練準確: {train_acc_t2:.1f}% | lr: {current_lr:.6f}")

        if train_acc_t2 > best_t2_acc:
            best_t2_acc = train_acc_t2
            torch.save(model.state_dict(), 'best_multitask_drsn.pth')

    print(f"\n🎉【訓練完畢！】最優 t2 訓練準確度：{best_t2_acc:.1f}%，模型已儲存。")
    print("=" * 60)

