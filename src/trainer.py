import logging
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset import MNISTDataModule
from src.model import BaseModel


def train(conf: dict) -> None:
    """Perform the training and save the weights

    Args:
        conf (dict, optional): Training conf. Defaults to None.
    """
    # Configuring logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(module)s - %(message)s'
    )

    # Path to save weights for this experiment
    dirpath = Path(conf["experiment"]["save_dir"],
                             conf["experiment"]["name"]

    # Loading the datamodule
    dm_mnist = MNISTDataModule(**conf["datamodule"])

    # Getting the model
    model = BaseModel(**conf["model"])

    # Setting up the trainer
    trainer = pl.Trainer(
        logger=TensorBoardLogger(**conf["experiment"]),
        callbacks=[ModelCheckpoint(dirpath=dirpath, **conf["checkpoints"])],
        **conf["trainer"]
    )

    # Training
    trainer.fit(model, datamodule=dm_mnist)


def test_trainer() -> None:
    # Reading conf file
    with open("conf/configuration.yaml", "r") as f:
        conf = yaml.safe_load(f)

    # Overriding conf for testing
    conf["trainer"]["max_epochs"] = 1
    conf["trainer"]["overfit_batches"] = 3

    # Testing trainer
    train(conf)
