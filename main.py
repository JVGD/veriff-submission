import yaml

from src.trainer import train


# Entrypoint from the docker container
# this files calls the trainer with
# the given conf mounted in /conf


if __name__ == "__main__":
    # Reading YAML conf (/conf needs to be mounted in Docker Container)
    with open("/conf/configuration.yaml", "r") as f:
        conf = yaml.safe_load(f)

    # Training
    train(conf)
