"""Trial-wise DeepConvNet for (B, 1, 2, 750) EEG trials.

Independent PyTorch implementation of the four-block architecture described by
Schirrmeister et al. (2017), https://arxiv.org/abs/1703.05051.
Temporal kernels 10, max-pooling size/stride 3; adapted two-class dense head.
"""
import math
import torch
from torch import nn


class DeepConvNet(nn.Module):
    def __init__(self, activation='elu', dropout=0.5, elu_alpha=1.0,
                 negative_slope=0.01, conv_max_norm=2.0, classifier_max_norm=0.5):
        super().__init__()
        if activation not in ('elu','relu','leaky_relu'):
            raise ValueError('Unknown activation')
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('dropout must be finite and in [0, 1)')
        if not math.isfinite(elu_alpha) or elu_alpha<=0 or not math.isfinite(negative_slope) or negative_slope<0:
            raise ValueError('Invalid activation parameters')
        if any(not math.isfinite(v) or v<=0 for v in (conv_max_norm,classifier_max_norm)):
            raise ValueError('Max norms must be finite and positive')
        self.conv_max_norm=conv_max_norm
        self.classifier_max_norm=classifier_max_norm
        def post(channels):
            layer={'elu':lambda:nn.ELU(elu_alpha),'relu':nn.ReLU,
                   'leaky_relu':lambda:nn.LeakyReLU(negative_slope)}[activation]()
            return [nn.BatchNorm2d(channels,eps=1e-5,momentum=0.1),layer,
                    nn.MaxPool2d((1,3),stride=(1,3)),nn.Dropout(dropout)]
        self.block1=nn.Sequential(nn.Conv2d(1,25,(1,10),bias=False),
                                  nn.Conv2d(25,25,(2,1),bias=False),*post(25))
        self.block2=nn.Sequential(nn.Conv2d(25,50,(1,10),bias=False),*post(50))
        self.block3=nn.Sequential(nn.Conv2d(50,100,(1,10),bias=False),*post(100))
        self.block4=nn.Sequential(nn.Conv2d(100,200,(1,10),bias=False),*post(200))
        self.classify=nn.Linear(200*4,2)
        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None: nn.init.zeros_(layer.bias)
        self.constrain_weights()

    @torch.no_grad()
    def constrain_weights(self):
        """Project each output filter/row after every optimizer update."""
        for layer in self.modules():
            if isinstance(layer,(nn.Conv2d,nn.Linear)):
                limit=self.conv_max_norm if isinstance(layer,nn.Conv2d) else self.classifier_max_norm
                flat=layer.weight.flatten(1)
                norms=flat.norm(2,dim=1,keepdim=True).clamp_min(1e-12)
                scale=(limit/norms).clamp_max(1)
                layer.weight.mul_(scale.reshape(-1,*([1]*(layer.weight.ndim-1))))

    def forward(self,inputs):
        if inputs.ndim!=4 or tuple(inputs.shape[1:])!=(1,2,750):
            raise ValueError('Expected input shape (B, 1, 2, 750)')
        features=self.block4(self.block3(self.block2(self.block1(inputs))))
        return self.classify(features.flatten(1))
