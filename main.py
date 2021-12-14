import pytorch_lightning as pl
import yaml

from src.tester import test
from src.trainer import train

# Entrypoint from the docker container
# this files calls the trainer with
# the given conf mounted in /conf


if __name__ == "__main__":
    # Reproducibility
    pl.seed_everything(42, workers=True)

    # Reading YAML conf (/conf needs to be mounted in Docker Container)
    with open("./conf/configuration.yaml", "r") as f:
        conf = yaml.safe_load(f)

    # Phase action defined by conf either train or test
    if conf["experiment"]["phase"] == 'train':
        # Run trainer
        train(conf)
    elif conf["experiment"]["phase"] == 'test':
        # Run tester
        test(conf)
    else:
        raise ValueError(
            "Only supported phases are: train and test however phase {} "
            "was given, review configuration with path /experiment/phase "
            "in YAML".format(conf["experiment"]["phase"])
        )