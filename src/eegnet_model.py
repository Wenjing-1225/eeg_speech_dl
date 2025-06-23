#!/usr/bin/env python
# eegnet_model.py
# -----------------------------------------------------------
# 纯粹的 EEGNet 定义：可被任何脚本 import，而不会自动运行
# -----------------------------------------------------------
import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """
    Minimal 3-block EEGNet (Lawhern et al., 2018) 适配 2-秒窗口。
    参数:
        C (int):  输入通道数
        T (int):  每个窗口时间点数 (samples)
        n_cls (int): 分类类别数
    """
    def __init__(self, C: int, T: int, n_cls: int = 2):
        super().__init__()

        # Block-1: Temporal conv
        self.conv1 = nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False)
        self.bn1   = nn.BatchNorm2d(8)

        # Block-2: Depthwise spatial conv
        self.conv2 = nn.Conv2d(8, 16, (C, 1), groups=8, bias=False)
        self.bn2   = nn.BatchNorm2d(16)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(0.25)

        # Block-3: Separable conv
        self.conv3 = nn.Conv2d(16, 16, (1, 16), padding=(0, 8), bias=False)
        self.bn3   = nn.BatchNorm2d(16)
        self.pool3 = nn.AvgPool2d((1, 8))
        self.drop3 = nn.Dropout(0.25)

        # 计算扁平化后特征长度
        out_len = ((T + 64 - 1) // 1 - 63) // 4   # after pool2
        out_len = ((out_len + 16 - 1) // 1 - 15) // 8  # after pool3
        self.fc = nn.Linear(16 * out_len, n_cls)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:  x  [B, 1, C, T]
        输出:  logits [B, n_cls]
        """
        x = torch.relu(self.bn1(self.conv1(x)))

        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.drop2(x)

        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.drop3(x)

        x = x.flatten(1)        # [B, features]
        return self.fc(x)