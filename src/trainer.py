import logging
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.callbacks import DrawSTNTransform
from src.dataset import MNISTDataModule
from src.stnmodel import STNModel
from src.coordconvmodel import CoordConvModel


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
    dirpath = Path(conf["experiment"]["save_dir"], conf["experiment"]["name"])
    logging.info(f"saving weights: {dirpath.absolute()}")

    # Loading the datamodule
    dm_mnist = MNISTDataModule(**conf["datamodule"])

    # Getting the model
    model = load_model(conf)

    # Setting up the trainer
    trainer = pl.Trainer(
        logger=TensorBoardLogger(default_hp_metric=False, **conf["experiment"]),
        callbacks=[
            ModelCheckpoint(dirpath=dirpath, **conf["checkpoints"]),
            EarlyStopping(**conf["early_stopping"]),
            DrawSTNTransform()
        ],
        deterministic=True,
        **conf["trainer"]
    )

    # Training
    trainer.fit(model, datamodule=dm_mnist)


def load_model(conf: dict) -> pl.LightningModule:
    """Loads the appropriate model from available ones

    Returns vanilla model or trained model according
    to phase defined in conf: train or test

    Args:
        conf (dict): Configuration dict

    Returns:
        pl.LightningModule: Model to return
    """
    # Available models
    available_models = {
        "STNModel": STNModel,
        "CoordConvModel": CoordConvModel
    }

    # Getting model name & sanity check
    model_name = conf["model"].pop('name')
    assert model_name in available_models, "Model selected does not exist"
    logging.info(f"Model: {model_name}")

    # Getting model class
    Model = available_models[model_name]

    # Getting model instance
    if conf["phase"] == 'train':
        # Instance for training
        model = Model(**conf["model"])

    if conf["phase"] == 'test':
        # Instance from checkpoint for testing
        model = STNModel.load_from_checkpoint(conf["tester"]["checkpoint"])

    return model


def test_trainer() -> None:
    # Reading conf file
    with open("./conf/configuration.yaml", "r") as f:
        conf = yaml.safe_load(f)

    # Overriding conf for testing
    conf["trainer"]["max_epochs"] = 1
    conf["trainer"]["overfit_batches"] = 3
    conf["early_stopping"]["monitor"] = "Loss/Train"

    # Testing trainer
    train(conf)
