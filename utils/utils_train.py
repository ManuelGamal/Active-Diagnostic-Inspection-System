import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiProjectionLayer(nn.Module):
    """Multi-scale feature projection layer from RD++."""

    def __init__(self, base=64):
        super().__init__()
        self.proj_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base, kernel_size=1),
                nn.BatchNorm2d(base),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(512, base, kernel_size=1),
                nn.BatchNorm2d(base),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(1024, base, kernel_size=1),
                nn.BatchNorm2d(base),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(2048, base, kernel_size=1),
                nn.BatchNorm2d(base),
                nn.ReLU(inplace=True),
            ),
        ])

    def forward(self, features):
        projected = []
        for i, block in enumerate(self.proj_blocks):
            projected.append(block(features[i]))
        return projected
