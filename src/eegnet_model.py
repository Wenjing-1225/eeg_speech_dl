#!/usr/bin/env python
# eegnet_model.py  ——  自动确定 fc 输入维度的 EEGNet
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, C: int, T: int, n_cls: int = 2):
        super().__init__()
        # ------------ Block-1 -------------
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.bn1   = nn.BatchNorm2d(8)

        # ------------ Block-2 -------------
        self.conv2 = nn.Conv2d(8, 16, (C, 1), groups=8, bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(0.25)

        # ------------ Block-3 -------------
        self.conv3 = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False)
        self.bn3   = nn.BatchNorm2d(16)
        self.pool3 = nn.AvgPool2d((1, 8))
        self.drop3 = nn.Dropout(0.25)

        # ----------- 自动推断 fc 尺寸 -----------
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)       # [B,1,C,T]
            feat  = self._forward_features(dummy) # 展平前
            in_dim = feat.shape[1]                # e.g., 640
        self.fc = nn.Linear(in_dim, n_cls)

    # -------- 把卷积 & 池化统一放到一个函数里 --------
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)
        return x.flatten(1)

    # ------------------- 前向 -------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        return self.fc(x)