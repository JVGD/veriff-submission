import logging
import sys
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append(".")
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

    # Loading the datamodule
    dm_mnist = MNISTDataModule(**conf["datamodule"])

    # Getting the model
    model = BaseModel(**conf["model"])

    # Setting up the trainer
    trainer = pl.Trainer(
        logger=TensorBoardLogger(**conf["experiment"]),
        callbacks=[
            ModelCheckpoint(
                dirpath=Path(conf["experiment"]["save_dir"],
                             conf["experiment"]["name"]
                ),
                **conf["checkpoints"]
            )
        ],
        **conf["trainer"]
    )

    # Training
    trainer.fit(model, datamodule=dm_mnist)


if __name__ == "__main__":
    import yaml
    with open("./conf/configuration.yaml", "r") as f:
        conf = yaml.safe_load(f)
    train(conf)
