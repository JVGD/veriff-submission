import torch.nn.functional as F
from torch import nn

from src.sk import SKAttention
from src.stnmodel import STNModel


class SKModel(STNModel):
    """SK Model

    Same as base model STNModel but replacing
    conv by coord conv the rest is the same
    """
    def __init__(self, optimizer: dict) -> None:
        """Init CoordConvModel from STNModel setup

        Args:
            optimizer (dict): Dict with optimizer conf
        """
        # Init base class
        super().__init__(optimizer)

        # Overwriting conv with Conv + SK Attention
        self.conv1 = nn.ModuleList([
            nn.Conv2d(1, 10, kernel_size=5),
            SKAttention(10)
        ])
        self.conv2 = nn.ModuleList([
            nn.Conv2d(10, 20, kernel_size=5),
            SKAttention(20)
        ])
