import torch as T
import torch.nn.functional as F
from torch import nn


class SpatialTransformerNetwork(nn.Module):
    """Spatial Transformer Network Module

    Implementation of the Deep Mind paper:
    `STN <https://arxiv.org/abs/1506.02025>`_
    """
    def __init__(self) -> None:
        # Init base class
        super().__init__()

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor layer for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            T.tensor([1, 0, 0, 0, 1, 0], dtype=T.float)
        )

    def forward(self, x: T.Tensor) -> T.Tensor:
        """STN Forward Pass"""
        # Section 3.1 in the paper: localization network
        # Conv Net: from [B,1,H,W] to [B,10,H',W']
        xs = self.localization(x)

        # Regressor: from [B,10,H',W'] to [B,2,3]
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Section 3.2 in the paper
        grid = F.affine_grid(theta, x.size())

        # Section 3.3 in the paper
        x = F.grid_sample(x, grid)
        return x


def test_SpatialTransformerNetwork() -> None:
    """Testing STN inference and shapes"""
    # Testing instantiation
    model = SpatialTransformerNetwork()

    # Random input
    x = T.rand(8, 1, 28, 28)
    y = model(x)
    assert y.shape == (8, 1, 28, 28)