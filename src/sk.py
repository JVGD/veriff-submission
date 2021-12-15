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
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=T.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=T.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V






if __name__ == '__main__':
    input=T.randn(50,10,28,28)
    se = SKAttention(channel=10,reduction=10)
    output=se(input)
    print(output.shape)