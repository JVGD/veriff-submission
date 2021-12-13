import logging

import pytorch_lightning as pl
import torch as T
import torch.nn.functional as F
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData

from src.stn import SpatialTransformerNetwork


class BaseModel(pl.LightningModule):
    """Base Model

    This PL module implements a base convolutional model which has
    been upgraded with a STN. Additional it includes extra logic that
    helps to abstract the engineering code from the research code.
    For more info read the pytorch lightning docs:

    `PL Docs <https://pytorch-lightning.rtfd.io/en/latest/>`_
    """
    def __init__(self, optimizer: dict) -> None:
        """Init model and base layers

        Args:
            optimizer (dict): Dict with optimizer conf
        """
        # Init base class
        super().__init__()

        # Configuration for optimizer
        self.optimizer_conf = optimizer

        # STN
        self.stn = SpatialTransformerNetwork()

        # Base Model
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Logging
        logging.info(f"{self.__class__.__name__}")
        logging.info(f"optimizer: {self.optimizer_conf}")

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

    def configure_optimizers(self) -> T.optim.Optimizer:
        """Optimizer to use during training"""
        return T.optim.SGD(self.parameters(), **self.optimizer_conf)

    def training_step(self, batch, batch_idx) -> T.Tensor:
        """Training step for the training loop"""
        # Unpacking batch, forward, loss and logging
        samples, targets_true = batch
        targets_pred = self(samples)
        loss = F.nll_loss(input=targets_pred, target=targets_true)
        self.log("loss_train", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        """Validation step for the validation loop"""
        # Unpacking batch, forward, loss and logging
        samples, targets_true = batch
        targets_pred = self(samples)
        loss = F.nll_loss(input=targets_pred, target=targets_true)
        self.log("loss_valid", loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx) -> None:
        """Test step for the test loop"""
        # Unpacking batch, forward, loss and logging
        samples, targets_true = batch
        targets_pred = self(samples)
        loss = F.nll_loss(input=targets_pred, target=targets_true)
        self.log("loss_test", loss, on_step=False, on_epoch=True)


def test_BaseModel_forward() -> None:
    """Testing base model"""
    # Testing instancing
    model = BaseModel(optimizer={"lr": 0.01})

    # Testing inference shapes
    x = T.rand(8,1,28,28)
    y = model(x)
    assert y.shape == (8, 10)


def test_BaseModel_train() -> None:
    """Testing training with fake data"""
    # Fake dataset we are testing training logic not convergence
    ds = FakeData(
        size=4*5,
        image_size=(1, 28, 28),
        num_classes=8,
        transform=transforms.Compose([transforms.ToTensor()])
    )

    # Dataloaders
    dl_train = DataLoader(dataset=ds, batch_size=4, shuffle=False)
    dl_valid = DataLoader(dataset=ds, batch_size=4, shuffle=False)

    # Model to test
    model = BaseModel(optimizer={"lr": 0.01})

    # Trainer setups for test
    trainer = pl.Trainer(
        logger=False,
        checkpoint_callback=False,
        enable_progress_bar=True,
        overfit_batches=5,
        # fast_dev_run=True,
        max_epochs=1,
        accelerator="cpu"
    )

    # Training test: model + data
    trainer.fit(model, train_dataloaders=dl_train, val_dataloaders=dl_valid)