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

To provide the conf to the container the conf file in your host system will be maped into /conf/ inside the container as a volume (for more info read the makefile).

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

## Research

Tensorboard with graphs, losses, metrics and final scores is published here: TODO

The findings will be explained in a live session but we will outline main results here: [Tensorboard](https://tensorboard.dev/experiment/sNygezqtRc6tornRzFeApg/). **NOTE: Uploading images to Tensorboard dev is yet not supported so STN images will be uploaded here**

* STN: It can effectively transform the input image to sample the most representative parts of the image (see Tensorboard images)
* CoordConv: By adding coordinate channels to the input image it eases the operations of spatial transformations
* SK: Uses selective attention based on multiple scale input allowing the CNN to modify its input receptive field dynamically 

![STN Transforms](https://user-images.githubusercontent.com/6736158/146327856-eef91d2f-30ff-4f67-9436-5d07ea7c096a.png)

Two metrics have been considered: Precision and Accuracy. The justification is as follows:

* Precision: if we have a client autheticating for getting a loan we do not want to have False Positives (FP) because the cost of giving a loan to the wrong person is very high. On the other hand, if the rightful holder of the identity fails to authenticate, a False Negative (FN), although annoying, he can try again and succeed. So the cost of the FN is not as high and can be tolerable.

* Accuracy: for clients with a different revenue model, for example authenticating the age of the user to get a ride in an electric motorbike. The electric motorbike company wants as many rides as possible. It does not want to lend motorbikes to underage people but it does not care if we miss few predictions from time to time because that turns into revenue.

![Precision](https://user-images.githubusercontent.com/6736158/146328354-4ea88035-8611-44e9-85ea-bb779fc32520.png)


## Development

Development features:

* Packaging: projet packaged as Docker container
* Usage: through Makefile interface & conf files (scalable to cluster instead of scripting args)
* Automated testing: test driven development and automated testing with `pytest`
* Coding: created a simple python module under `./src` and python dependencies are tracked in `requirements.txt`
* Reproducibility: seeded everything for exact reproducible results

The project structure is as follows:

```
├── Dockerfile: Packages project with reproducible image
├── MNIST: Data that will be downloaded
├── Makefile: Package most common actions: run, test and dev
├── conf: Configuration folder that will be mounted into the docker as a volume
│   └── configuration.yaml: Configuration file for this project
├── main.py: Entrypoint for the docker image
├── pytest.ini: Configuration for pytest, automated code testing
├── requirements.txt: Python dependencies
├── src: code
│   ├── __init__.py: Python module declaration
│   ├── callbacks.py: Callbacks for drawing digits in tensorboard
│   ├── coordconv.py: CoordConv module in pytorch
│   ├── coordconvmodel.py: CoorConv model for training & testing
│   ├── dataset.py: DataModule from PL: dataset + datamodule
│   ├── stn.py: STN module in pytorch
│   ├── stnmodel.py: STN model for training & testing
│   ├── tester.py: tester script
│   └── trainer.py: trainer script
└── weights: automatically created when training, store weights
    ├── BaseModel: experiment folder named after the experiment name
    └── CoordConvModel: : experiment folder named after the experiment name
```



