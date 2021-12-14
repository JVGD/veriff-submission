import sys

import pytorch_lightning as pl

sys.path.append(".")
from src.dataset import MNISTDataModule
from src.model import BaseModel


def test(conf: dict) -> None:
    """Tester

    Args:
        conf (dict): Configuration
    """
    # Loading the datamodule
    dm_mnist = MNISTDataModule(**conf["datamodule"])

    # Loading model from checkpoint
    model = BaseModel.load_from_checkpoint("weights/BaseModel-001-Test-Ckpt/epoch=2-step=8.ckpt")

    # Loading PL engine
    trainer = pl.Trainer(deterministic=True)

    # Running test loop and getting a metric
    trainer.test(model, datamodule=dm_mnist)


if __name__ == "__main__":
    import yaml

    # Reading YAML conf (/conf needs to be mounted in Docker Container)
    with open("./conf/configuration.yaml", "r") as f:
        conf = yaml.safe_load(f)

    # Training
    test(conf)
