"""
A dummy network named RFSNet, for video crowd counting
"""

import torch.nn as nn


class RFSNet(nn.Module):
    def __init__(self, cfg):
        super(RFSNet, self).__init__()

        self.Encoder = nn.Identity()

    def forward(self, x):
        x = self.Encoder(x)
        return x
