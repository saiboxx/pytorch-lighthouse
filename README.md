# Pytorch Lighthouse

*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

*~~~~~~~~~~ PyTorch Lightning + Docker template for smooth sailing ~~~~~~~~~~*

*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*


This project forms a template that enables fast prototyping with Pytorch 
Lightning and deployment with ease via docker.

## How to use

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