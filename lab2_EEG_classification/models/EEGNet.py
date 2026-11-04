"""EEGNet for two-channel, 750-sample EEG trials."""
import math
import torch
from torch import nn


def activation_layer(name, alpha=1.0, negative_slope=0.01):
    if name == "elu":
        return nn.ELU(alpha=alpha)
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=negative_slope)
    raise ValueError(f"Unknown activation: {name}")


class EEGNet(nn.Module):
    """Slide architecture by default; optional true depthwise-separable block.

    The slide's 'separableConv' is actually a regular temporal Conv2d.
    temporal_block='depthwise_separable' replaces it with depthwise + pointwise
    convolution. Both produce 736 features for input (B, 1, 2, 750).
    """

    def __init__(self, activation="elu", elu_alpha=1.0, dropout=0.25,
                 negative_slope=0.01, temporal_block="slide"):
        super().__init__()
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if (not math.isfinite(elu_alpha) or not math.isfinite(negative_slope)
                or elu_alpha <= 0 or negative_slope < 0):
            raise ValueError("ELU alpha must be finite and positive; negative slope must be finite and nonnegative")
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, (1, 51), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16),
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, (2, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            activation_layer(activation, elu_alpha, negative_slope),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        if temporal_block == "slide":
            temporal_layers = [nn.Conv2d(32, 32, (1, 15), padding=(0, 7), bias=False)]
        elif temporal_block == "depthwise_separable":
            temporal_layers = [
                nn.Conv2d(32, 32, (1, 15), padding=(0, 7), groups=32, bias=False),
                nn.Conv2d(32, 32, (1, 1), bias=False),
            ]
        else:
            raise ValueError(f"Unknown temporal_block: {temporal_block}")
        self.separableConv = nn.Sequential(
            *temporal_layers,
            nn.BatchNorm2d(32),
            activation_layer(activation, elu_alpha, negative_slope),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.classify = nn.Linear(736, 2)

    def forward(self, x):
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 2, 750):
            raise ValueError(f"Expected (B, 1, 2, 750), got {tuple(x.shape)}")
        x = self.separableConv(self.depthwiseConv(self.firstconv(x)))
        return self.classify(torch.flatten(x, 1))  # Raw logits for cross entropy.
