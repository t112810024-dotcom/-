import torch
import torch.nn as nn
import torch.nn.functional as F

class ShrinkageBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ShrinkageBlock, self).__init__()
        self.downsample = (stride != 1 or in_channels != out_channels)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        if self.downsample:
            residual = self.shortcut(x)
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        abs_mean = self.gap(torch.abs(out)).view(out.size(0), out.size(1))
        scales = self.fc(abs_mean).view(out.size(0), out.size(1), 1, 1)
        thres = abs_mean.view(out.size(0), out.size(1), 1, 1) * scales
        
        out = torch.sign(out) * torch.relu(torch.abs(out) - thres)
        out += residual
        out = self.relu(out)
        return out

class MultiTaskDRSN(nn.Module):
    def __init__(self):
        super(MultiTaskDRSN, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = ShrinkageBlock(32, 64, stride=2)
        self.layer2 = ShrinkageBlock(64, 128, stride=2)
        self.layer3 = ShrinkageBlock(128, 256, stride=2)
        self.layer4 = ShrinkageBlock(256, 256, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.head_t1 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 2))
        self.head_t2 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 3))
        self.head_t3 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 8))
        self.head_t4 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 5))
        self.head_t5 = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).view(x.size(0), -1)
        return {
            'task1': self.head_t1(x), 'task2': self.head_t2(x),
            'task3': self.head_t3(x), 'task4': self.head_t4(x),
            'task5': self.head_t5(x).squeeze(-1)
        }

class AutoencoderGatekeeper(nn.Module):
    def __init__(self):
        super(AutoencoderGatekeeper, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(7, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 7, 3, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # 🎯 物理對齊：先通過解碼器，再強制利用插值對齊原圖的 H 與 W (解決 1201 奇數邊界)
        recon = self.decoder(self.encoder(x))
        return F.interpolate(recon, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)