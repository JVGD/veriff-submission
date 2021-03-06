import sys

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.append(".")
from src.dataset import MNISTDataModule
from src.stnmodel import STNModel
from src.trainer import load_model


def check(conf: dict) -> None:
    """Tester

    Args:
        conf (dict): Configuration
    """
    # Loading the datamodule
    dm_mnist = MNISTDataModule(**conf["datamodule"])

    # Loading model from checkpoint
    model = load_model(conf)

    # Loading PL engine
    trainer = pl.Trainer(
        deterministic=True,
        logger=TensorBoardLogger(
            default_hp_metric=False,
            version="test",
            **conf["experiment"]
        )
    )

    # Running test loop and getting a metric
    trainer.test(model, datamodule=dm_mnist)