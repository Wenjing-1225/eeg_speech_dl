#!/usr/bin/env python
# ResNet-18-EEG —— 适配 [B,1,C,T]，电极维度保持不变
import torch
import torch.nn as nn


# ---------- 残差块 ----------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride_t=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, (1, 3),
                               stride=(1, stride_t),
                               padding=(0, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch, momentum=0.2)

        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, 3),
                               padding=(0, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch, momentum=0.2)

        self.short = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1,
                          stride=(1, stride_t), bias=False),
                nn.BatchNorm2d(out_ch, momentum=0.2)
            ) if (in_ch != out_ch or stride_t != 1) else nn.Identity()
        )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.short(x))


# ---------- ResNet-18-EEG ----------
class ResNetEEG(nn.Module):
    def __init__(self, C: int, T: int, n_cls: int = 2,
                 cfg=(16, 32, 64, 128)):
        super().__init__()
        self.C = C   # 保存电极数

        self.stem = nn.Sequential(
            nn.Conv2d(1, cfg[0], (1, 7), padding=(0, 3), bias=False),
            nn.BatchNorm2d(cfg[0], momentum=0.2),
            nn.ReLU(inplace=True)
        )

        self.layer1 = self._make_layer(cfg[0], cfg[0], stride_t=1)
        self.layer2 = self._make_layer(cfg[0], cfg[1], stride_t=2)
        self.layer3 = self._make_layer(cfg[1], cfg[2], stride_t=2)
        self.layer4 = self._make_layer(cfg[2], cfg[3], stride_t=2)

        # 自动推断 fc 输入维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            feat  = self._forward_features(dummy)
        self.fc = nn.Linear(feat.shape[1], n_cls)

    def _make_layer(self, in_ch, out_ch, stride_t):
        return nn.Sequential(
            BasicBlock(in_ch,  out_ch, stride_t),
            BasicBlock(out_ch, out_ch, 1)
        )

    def _forward_features(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 电极维做 global average；时域再 /4
        x = nn.AvgPool2d((self.C, 4))(x)
        return x.flatten(1)

    def forward(self, x):
        x = self._forward_features(x)
        return self.fc(x)