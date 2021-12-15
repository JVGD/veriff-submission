import torch as T
from torch import nn
from collections import OrderedDict


class SKAttention(nn.Module):
    """Selective Kernel Attention Module

    From paper: `SKNets <https://arxiv.org/abs/1903.06586>`_
    """
    def __init__(self,
        channel: int,
        kernels: list=[1,3,5],
        reduction: int=5,
        group: int=1,
        L: int=32
    ) -> None:
        """Init SK block

        Args:
            channel (int): Input channels.
            kernels (list, optional): Kernels for splitting branches.
            reduction (int, optional): Reduction ratio. Defaults to 16.
            group (int, optional): Depthwise conv. Defaults to 1.
            L (int, optional): Min size for d (z vector). Defaults to 32.
        """
        # Init base class
        super().__init__()

        # Computing d size
        self.d = max(L, channel // reduction)

        # Convolutions for splitting
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel, channel, kernel_size=k,
                                      padding=k//2,groups=group)
                    ),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )

        # Reduction layer
        self.fc = nn.Linear(channel, self.d)

        # Select layer: soft attention across channels
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))

        # Ouput softmax
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Forward pass of SK Block

        Args:
            x (T.Tensor): Input feature map

        Returns:
            T.Tensor: Output feature map of same size
        """
        # Getting input size
        B, C, _, _ = x.size()

        # Splitting: saving output of every branch
        conv_outs=[]
        for conv in self.convs:
            conv_outs.append(conv(x))

        # Stacking outputs of the splits
        feats = T.stack(conv_outs, 0)

        # Fuse
        U = sum(conv_outs)      # B x C x H x W

        # Reduction channel
        S = U.mean(-1).mean(-1) # B x C
        Z = self.fc(S)          # B x d

        # Selection or attention
        weights=[]
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(B ,C, 1, 1)) # B x C

        weights_att = T.stack(weights, 0)           # k x B x C x 1 x 1
        weights_att = self.softmax(weights_att)     # k x B x C x 1 x 1

        # Fusing
        V = (weights_att * feats).sum(0)
        return V


def test_SKBlock() -> None:
    """Testing SK Block"""
    x = T.randn(8, 10, 28, 28)
    se = SKAttention(channel=10,reduction=4)
    y = se(x)
    print(y.shape)
    assert y.shape == (8, 10, 28, 28)