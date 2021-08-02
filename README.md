# Pytorch Lighthouse

*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

*~~~~~~~~~~ PyTorch Lightning + Docker template for smooth sailing ~~~~~~~~~~*

*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*


This project forms a template that enables fast prototyping with Pytorch
Lightning and deployment with ease via docker.

## How to use

The `src` folder contains a few files, which need to be filled (depending on the
use case). Here the structure of the objects created follows the PyTorch or Lightning
API and therefore are not investigated further.

- `data.py` --> LightningDataModule, DataSet
- `module.py` --> LightningModule
- `network.py` --> PyTorch neural network

The trainer is part of the main loop in `main.py`.

Via the config file `config.yml` parameters are directly passed to the respective
modules, which enables a quick change of settings without editing source code.


## Conducting training

The only requirement is a valid docker install on your Linux system. It may be
possible that you also need the `nvidia-container-toolkit`.

First, the container is built:

```
make build-image
```

To execute the code that is specified in the entrypoint `main.py`, call:

```
make run-image
```

The project directory is mounted as a volume in the container, which means after
changing code the container does not need to be rebuild.

There is also a chance that there is a communication problem between the nvidia drivers
and the docker daemon. In this case the necessary nvidia devices need to be passed
manually, which is handled by the following call. This needs to be adjusted in a
multi-GPU setting.

```
make run-image-explicit
```

Tensorboard tracking is automatically enabled and forwarded. It is accessible on
`localhost:6006`. Logs will be saved in the `logs` directory.
