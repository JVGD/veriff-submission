import torch as T
from torch import nn


class AddCords(nn.Module):
    """Add 2 coordinate channels to input tensor

    From a tensor of `B x C x H x W` it transforms it to
    a tensor of `B x (C+2) x H x W` where C+2 is the adding
    of the `i` and `j` channels

    It is defined as a Pytorch nn Module for modularity
    although this operation is entierly functional

    Simplified 2D version inspired in code of section S8
    from paper: `CoordConv <https://arxiv.org/aB/1807.03247>`_
    """
    def __init__(self) -> None:
        """Defined as a Pytorch nn Module without any parameters"""
        super().__init__()

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Create coordiates i, j channels and concat to input tensor

        Args:
            x (T.Tensor): Input tensor to add coords info to

        Returns:
            T.Tensor: Tensor with additional i,j channel coords
        """
        # Getting input shape
        B, C, H, W = x.size()

        # Building coordiantes channels
        i = (T.range(start=0, end=H-1).unsqueeze(0)
                                      .unsqueeze(0)
                                      .unsqueeze(-1)
                                      .repeat([B,1,1,W])/(H/2)-1)
        j = (T.range(start=0, end=W-1).unsqueeze(0)
                                      .unsqueeze(0)
                                      .unsqueeze(0).
                                      repeat([B,1,H,1])/(W/2)-1)
        # Concatenation and return
        return T.cat([x, i, j], dim=1)


class CoordConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, *args, **kwargs) -> None:
        """CoordConv init

        Supports all the parameters from `nn.Conv2d` with
        argument expansion: `*args` and `**kwargs`

        Args:
            in_channels (int): Input channels to Conv
            out_channels (int): Out channels from Conv
            kernel_size (int): Kernel size of conv operation
        """
        # Init base class
        super().__init__()

        # CoordConv is coordinate channels adder + standar conv
        # where standar conv has to process 2 extra channels
        self.add_coord = AddCords()
        self.conv = nn.Conv2d(in_channels=in_channels+2,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              *args, **kwargs)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Forward of CoordConv: add coord + conv

        Args:
            x (T.Tensor): Input image or feature map

        Returns:
            T.Tensor: Output feature map
        """
        # Adding extra coordinate channels
        x = self.add_coord(x)

        # Standar conv
        x = self.conv(x)
        return x