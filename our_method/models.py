import torch
import torch.nn as nn

class AutoencoderGatekeeper(nn.Module):
    def __init__(self):
        super(AutoencoderGatekeeper, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 7, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        target_h, target_w = x.size(2), x.size(3)
        recon = self.decoder(self.encoder(x))
        recon = recon[:, :, :target_h, :target_w]
        return recon

class ShrinkageBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ShrinkageBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        abs_out = torch.abs(out)
        scales = self.gap(abs_out).view(abs_out.size(0), -1)
        thres = scales * self.fc(scales)
        thres = thres.view(thres.size(0), thres.size(1), 1, 1)

        out = torch.sign(out) * torch.relu(abs_out - thres)
        out += residual
        return self.relu(out)

class MultiTaskDRSN(nn.Module):
    def __init__(self):
        super(MultiTaskDRSN, self).__init__()
        self.conv1 = nn.Conv2d(7, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = ShrinkageBlock(32, 64, stride=2)
        self.layer2 = ShrinkageBlock(64, 128, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc_t1 = nn.Linear(128, 2)   # 0=正常, 1=異常
        self.fc_t2 = nn.Linear(128, 3)   # ✅ 0=正常, 1=氣泡, 2=閥門
        self.fc_t3 = nn.Linear(128, 8)   # ✅ 從5改為8：0=無, 1~7=BP位置
        self.fc_t4 = nn.Linear(128, 5)   # 0=無, 1~4=SV位置
        self.fc_t5 = nn.Linear(128, 1)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.gap(out).view(out.size(0), -1)

        return {
            'task1': self.fc_t1(out),
            'task2': self.fc_t2(out),
            'task3': self.fc_t3(out),
            'task4': self.fc_t4(out),
            'task5': self.fc_t5(out).squeeze(-1)
        }