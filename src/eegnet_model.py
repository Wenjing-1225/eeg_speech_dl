#!/usr/bin/env python
# eegnet_model.py ―― 自动确定 fc 输入维度的 EEGNet（调过 BN & Dropout）
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, C: int, T: int, n_cls: int = 2):
        super().__init__()
        # ------------ Block-1 -------------
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.bn1   = nn.BatchNorm2d(8,  momentum=0.25)   # ↑ momentum

        # ------------ Block-2 -------------
        self.conv2 = nn.Conv2d(8, 16, (C, 1), groups=8, bias=False)
        self.bn2   = nn.BatchNorm2d(16, momentum=0.25)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(0.40)                    # ↑ dropout

        # ------------ Block-3 -------------
        self.conv3 = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False)
        self.bn3   = nn.BatchNorm2d(16, momentum=0.25)
        self.pool3 = nn.AvgPool2d((1, 8))
        self.drop3 = nn.Dropout(0.40)

        # 自动推断 fc 尺寸
        with torch.no_grad():
            dummy = torch.zeros(1, 1, C, T)
            feat  = self._forward_features(dummy)
        self.fc = nn.Linear(feat.shape[1], n_cls)

    # 卷积+池化
    def _forward_features(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x))); x = self.pool2(x); x = self.drop2(x)
        x = torch.relu(self.bn3(self.conv3(x))); x = self.pool3(x); x = self.drop3(x)
        return x.flatten(1)

    def forward(self, x):
        return self.fc(self._forward_features(x))