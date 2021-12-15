# Veriff Submission: A STN journey

This repo correspond to a job application for Veriff. The documenation is divided in 2 parts:

* Usage: how to quick start up with this
* Research: where results are shown and findings are discussed.
* Development: where code & engineering is documented.

## Usage

### Building the project (Docker Image)
This project is packaged using a container approach (Docker) so before using it we need to build the docker image for it described in the `Dockerfile`. The basic commands for building, running and testing the project have been packaged in the `Makefile` for ease.

```bash
make build
```

Then we can run the code tests to check everything is ok

```bash
make test
```

### Running the project

This project defined-by-conf so everything it can be done: training different models, hyperparameters tunning and model testing can be done by modifying the configuration file in `./conf/configuration.yaml`. The configuration file is properly documented.

### Training

We can train by setting the configuration file `phase: train` and by selecting a model from available models. You can also configure other hyperparameters in the configuration file. When you are don or if you just one to use the default params just issue:

```bash
make run
```
