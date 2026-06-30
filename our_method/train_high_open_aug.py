import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models import MultiTaskDRSN


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..'))
TRAIN_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'dataset', 'dataset', 'train', 'processed')
BASE_WEIGHT_PATH = os.path.join(BASE_DIR, 'best_multitask_drsn.pth')
OUTPUT_WEIGHT_PATH = os.path.join(BASE_DIR, 'best_multitask_drsn_highopen.pth')
THRESHOLD_PATH = os.path.join(BASE_DIR, 'high_open_valve_threshold.npy')


def load_training_arrays():
    x_train = np.load(os.path.join(TRAIN_PROCESSED_DIR, 'x_cwt_images.npy'))
    if x_train.shape[-1] == 7:
        x_train = np.transpose(x_train, (0, 3, 1, 2))

    return (
        x_train.astype(np.float32),
        np.load(os.path.join(TRAIN_PROCESSED_DIR, 'y_t1.npy')).astype(np.int64),
        np.load(os.path.join(TRAIN_PROCESSED_DIR, 'y_t2.npy')).astype(np.int64),
        np.load(os.path.join(TRAIN_PROCESSED_DIR, 'y_t3.npy')).astype(np.int64),
        np.load(os.path.join(TRAIN_PROCESSED_DIR, 'y_t4.npy')).astype(np.int64),
        np.load(os.path.join(TRAIN_PROCESSED_DIR, 'y_t5.npy')).astype(np.float32),
    )


def build_high_open_augmented_set(x_train, y_t1, y_t2, y_t3, y_t4, y_t5, seed=20260701):
    rng = np.random.default_rng(seed)
    normal_idx = np.where(y_t2 == 0)[0]
    valve_idx = np.where(y_t2 == 2)[0]

    synthetic_x = []
    synthetic_t1 = []
    synthetic_t2 = []
    synthetic_t3 = []
    synthetic_t4 = []
    synthetic_t5 = []

    high_open_levels = np.array([94.0, 95.0, 96.0, 97.0, 98.0], dtype=np.float32)

    for valve_i in valve_idx:
        valve_location = int(y_t4[valve_i])
        if valve_location == 0:
            continue

        for health in high_open_levels:
            normal_i = int(rng.choice(normal_idx))

            # A high opening valve leak is modeled as a weak residual from a known
            # valve-fault pattern added to a normal operating sample. The blend
            # strength is monotonic with damage severity and uses only train labels.
            severity = (100.0 - float(health)) / 100.0
            alpha = np.clip(0.06 + severity * 2.8, 0.08, 0.25)
            sample = (1.0 - alpha) * x_train[normal_i] + alpha * x_train[valve_i]

            synthetic_x.append(sample.astype(np.float32))
            synthetic_t1.append(1)
            synthetic_t2.append(2)
            synthetic_t3.append(0)
            synthetic_t4.append(valve_location)
            synthetic_t5.append(float(health))

    if not synthetic_x:
        raise RuntimeError('No synthetic high-open samples were generated.')

    x_aug = np.concatenate([x_train, np.stack(synthetic_x)], axis=0)
    t1_aug = np.concatenate([y_t1, np.array(synthetic_t1, dtype=np.int64)])
    t2_aug = np.concatenate([y_t2, np.array(synthetic_t2, dtype=np.int64)])
    t3_aug = np.concatenate([y_t3, np.array(synthetic_t3, dtype=np.int64)])
    t4_aug = np.concatenate([y_t4, np.array(synthetic_t4, dtype=np.int64)])
    t5_aug = np.concatenate([y_t5, np.array(synthetic_t5, dtype=np.float32)])
    weak_aug = (t2_aug == 2) & (t5_aug >= 70.0) & (t5_aug < 100.0)

    return x_aug, t1_aug, t2_aug, t3_aug, t4_aug, t5_aug, weak_aug


def class_weight(labels):
    counts = np.bincount(labels)
    weights = [len(labels) / (len(counts) * max(int(count), 1)) for count in counts]
    return torch.tensor(weights, dtype=torch.float32)


def calibrate_high_open_threshold(model, x_train, y_t2, device):
    model.eval()
    valve_probs = []
    with torch.no_grad():
        for i in np.where(y_t2 == 0)[0]:
            inputs = torch.tensor(x_train[i], dtype=torch.float32).unsqueeze(0).to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs['task2'], dim=1).cpu().numpy()[0]
            valve_probs.append(float(probs[2]))

    threshold = float(np.quantile(np.array(valve_probs, dtype=np.float32), 0.99))
    np.save(THRESHOLD_PATH, np.array([threshold], dtype=np.float32))
    return threshold


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_train, y_t1, y_t2, y_t3, y_t4, y_t5 = load_training_arrays()
    x_aug, t1_aug, t2_aug, t3_aug, t4_aug, t5_aug, weak_aug = build_high_open_augmented_set(
        x_train, y_t1, y_t2, y_t3, y_t4, y_t5
    )

    dataset = TensorDataset(
        torch.tensor(x_aug, dtype=torch.float32),
        torch.tensor(t1_aug, dtype=torch.long),
        torch.tensor(t2_aug, dtype=torch.long),
        torch.tensor(t3_aug, dtype=torch.long),
        torch.tensor(t4_aug, dtype=torch.long),
        torch.tensor(t5_aug, dtype=torch.float32),
        torch.tensor(weak_aug, dtype=torch.bool),
    )
    loader = DataLoader(dataset, batch_size=12, shuffle=True)

    model = MultiTaskDRSN().to(device)
    model.load_state_dict(torch.load(BASE_WEIGHT_PATH, map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35)

    criterion_t1 = nn.CrossEntropyLoss(reduction='none')
    criterion_t2 = nn.CrossEntropyLoss(weight=class_weight(t2_aug).to(device), reduction='none')
    criterion_t3 = nn.CrossEntropyLoss(weight=class_weight(t3_aug).to(device), reduction='none')
    criterion_t4 = nn.CrossEntropyLoss(weight=class_weight(t4_aug).to(device), reduction='none')
    criterion_t5 = nn.L1Loss(reduction='none')

    best_loss = float('inf')
    print(f'Augmented samples: original={len(x_train)}, total={len(x_aug)}, synthetic={len(x_aug) - len(x_train)}')

    for epoch in range(35):
        model.train()
        total_loss = 0.0
        for inputs, t1, t2, t3, t4, t5, weak_mask in loader:
            inputs = inputs.to(device)
            t1, t2, t3, t4, t5 = t1.to(device), t2.to(device), t3.to(device), t4.to(device), t5.to(device)
            weak_mask = weak_mask.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss_t1 = criterion_t1(outputs['task1'], t1)
            loss_t2 = criterion_t2(outputs['task2'], t2)
            loss_t3 = criterion_t3(outputs['task3'], t3)
            loss_t4 = criterion_t4(outputs['task4'], t4)
            loss_t5 = criterion_t5(outputs['task5'], t5)

            loss_t2 = torch.where(weak_mask, loss_t2 * 4.0, loss_t2).mean()
            loss_t4 = torch.where(weak_mask, loss_t4 * 2.0, loss_t4).mean()
            loss_t5 = torch.where(weak_mask, loss_t5 * 4.0, loss_t5).mean()

            loss = loss_t1.mean() + loss_t2 + loss_t3.mean() + loss_t4 + 0.2 * loss_t5
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        scheduler.step()
        avg_loss = total_loss / max(len(loader), 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), OUTPUT_WEIGHT_PATH)

        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1:02d}/35 | loss={avg_loss:.4f}')

    model.load_state_dict(torch.load(OUTPUT_WEIGHT_PATH, map_location=device))
    threshold = calibrate_high_open_threshold(model, x_train, y_t2, device)
    print(f'Saved high-open fine-tuned weights: {OUTPUT_WEIGHT_PATH}')
    print(f'Saved high-open valve threshold: {THRESHOLD_PATH} ({threshold:.6f})')


if __name__ == '__main__':
    main()
