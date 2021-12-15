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

#### Training

We can train by setting the configuration file `phase: train` and by selecting a model from available models. 

```yaml
phase: train            # 'train' or 'test', action to perform

model:
  name: CoordConvModel  # Model to use 'STNModel' or 'CoordConvModel'
  optimizer:
    lr: 0.01            # Learning rate
    momentum: 0         # SGD momentum
    weight_decay: 0     # SGD weight decay

trainer:
  max_epochs: 80              # Max number of epochs to run
  accelerator: cpu            # 'cpu', 'gpu', 'ipu'... according to HW
  gpus: null                  # Total number of GPUs to use
  tpu_cores: null             # Total number of TPUs to use
  ipus: null                  # Total number of IPUs to use
  precision: 32               # '32' or '16' for half precision
```

You can also configure other hyperparameters in the configuration file. When you are don or if you just one to use the default params just issue:

```bash
make run
```

Weights and logs will be saved in `/weights` inside the container that by default maps into `$PWD/weights` in your host system.

#### Testing

For testing we need to select the phase `phase: train` the target model and the path to the checkpoint file (weights).

```yaml
phase: test             # 'train' or 'test', action to perform

model:
  name: STNModel        # Model to use 'STNModel' or 'CoordConvModel'

tester:
  checkpoint: weights/BaseModel/epoch=60-step=52459.ckpt  # Path to checkpoint

```

Then run the project by issuing:

```bash
make run
```


