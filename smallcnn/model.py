"""The small CNN of Srivastava, Yang & Eckstein (2024) for joint training on all three tasks.

From the paper's joint-training notebook (One_Network_All_Three_Tasks5VS.ipynb, cell 91), Keras:
Conv(12, 3x3, stride 2) -> Conv(14, 3x3, stride 2) -> Conv(18, 2x2), all ReLU, no padding ->
BatchNorm -> Flatten -> Dense(600, ReLU) -> Dense(6). Input (N, 1, 224, 224) in 0-1. Trained
from scratch; Keras default init (Glorot uniform, zero bias) and BatchNorm settings.
Layer indices (forward(upto=i) returns the output of self.layers[i]): 0-2 conv1-3, 3 batch norm,
4 flatten, 5 Dense(600) (the units of lesion.py and neuron_cueing.py), 6 output.
"""
import torch
from torch import nn

DENSE = 5


class NHWCFlatten(nn.Module):
    def forward(self, x):  # Keras (H, W, C) order, so Dense weights line up with a Keras model's
        return x.permute(0, 2, 3, 1).flatten(1)


class JointCNN(nn.Module):
    def __init__(self, n_classes: int = 6, size: int = 224):
        super().__init__()
        side = ((size - 3) // 2 + 1 - 3) // 2 + 1 - 1
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, 12, 3, stride=2), nn.ReLU()),
            nn.Sequential(nn.Conv2d(12, 14, 3, stride=2), nn.ReLU()),
            nn.Sequential(nn.Conv2d(14, 18, 2), nn.ReLU()),
            nn.BatchNorm2d(18, eps=1e-3, momentum=0.01),  # Keras BatchNormalization defaults
            NHWCFlatten(),
            nn.Sequential(nn.Linear(side * side * 18, 600), nn.ReLU()),
            nn.Linear(600, n_classes),  # logits
        ])
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, upto: int | None = None):
        """Logits, or the output of self.layers[upto] (NCHW for the conv layers)."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == upto:
                return x
        return x
