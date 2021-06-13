import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Attn_block(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.ic = input_channels
        self.cnv = nn.Conv2d(self.ic, self.ic, 3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        y = x.clone()
        x = self.cnv(x)
        x = self.relu(x)
        x = self.pool(x)
        attn_wts = F.softmax(x, dim=1)
        op = y * attn_wts

        return op, attn_wts

if __name__ == "__main__":
    ip = torch.randn(1,3,20,20)
    attn = Attention_block(3)
    a, b = attn(ip)
    print(a.shape)
    print(b.shape)
    
    



