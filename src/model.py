import pytorch_lightning as pl
import torch as T
import torch.nn.functional as F
from torch import nn

from src.stn import SpatialTransformerNetwork


class BaseModel(pl.LightningModule):
    """Base Model

    This PL module implements a base convolutional model which has
    been upgraded with a STN. Additional it includes extra logic that
    helps to abstract the engineering code from the research code.
    For more info read the pytorch lightning docs:

    `PL Docs <https://pytorch-lightning.rtfd.io/en/latest/>`_
    """
    def __init__(self) -> None:
        # Init base class
        super().__init__()

        # STN
        self.stn = SpatialTransformerNetwork()

        # Base Model
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Forward Pass of the model"""
        # Spatially transform the input
        x = self.stn(x)

        # Do the forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def test_BaseModel() -> None:
    """Testing base model"""
    # Testing instancing
    model = BaseModel()

    # Testing inference shapes
    x = T.rand(8,1,28,28)
    y = model(x)
    assert y.shape == (8, 10)