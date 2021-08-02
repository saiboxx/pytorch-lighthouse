"""Classes and functions for creating Lightning modules."""
from typing import (
    Any,
    Dict,
    List,
)

import pytorch_lightning as pl
from torch import Tensor

from network import LitNetwork


class LitModule(pl.LightningModule):
    """
    A general lightning module.

    This template is basic and barebones. Lightning offers a variety of customization
    options and additional functions/parameters.
    Also the typing for a few parameters and return values depend heavily on your own
    implementation. Adjust to personal need.
    The full LightningModule API reference is here:
    https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html
    """

    def __init__(self, param: Any = None):
        """Initialize a lightning module."""
        super().__init__()
        self.param = param
        self.net = LitNetwork()

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward method for use in inference.

        :param x: Input tensor.
        :return: Output tensor.
        """
        return self.net(x)

    def configure_optimizers(self) -> Any:
        """
        Create optimizers.

        Specifies what optimizers and learning-rate schedulers to use in the
        optimization.
        The return can be a variety of options including multiple optimizers and
        learning schedulers. Thus, it is left on `Any` for now but should change
        according to the implementation. It is recommended to investigate the options
        in the official docs.
        """
        pass

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Conduct a training step.

        Here you compute and return the training loss and some additional metrics for
        e.g. the progress bar or logger. The training logic needs to be implemented in
        this method. The input is a batch of training data supplied by the Dataloader.
        The type of `batch` is dependent on your implementation but usually is a dict
        with Tensors. Please adjust the typing accordingly.
         Further, if multiple optimizers are used, the `training_step`
        method gets an additional parameter `optimizer_idx` of type `int` that indicates
        the current active optimizer, e.g. this method is called twice for two
        optimizers but with different indices. This behavior can be bypassed by turning
        off automatic backwarding (cf.
        https://github.com/PyTorchLightning/pytorch-lightning#pro-level-control-of-training-loops-advanced-users.)
        Return the loss tensor to perform automatic backwarding + updating.
        The return could also be a dict but it needs a key with `loss`.

        :param batch: Batch of training data.
        :param batch_idx: Index of the batch.
        :return: Loss tensor.
        """
        pass

    def training_step_end(self, training_step_outputs: List) -> Tensor:
        """
        Perform an action after a training step completes.

        Mostly relevant for multi-GPU training as the `training_step` method receives
        only a fraction (1/num_gpus) of the batch. If something should be done or
        calculated on the full batch, this gives the opportunity. The outputs of
        `training_step` are collected in a list and are passed.
        """
        pass

    def training_epoch_end(self, outputs: Any) -> None:
        """
        Perform an action after a training epoch completes.

        Called at the end of the training epoch with the outputs of all training steps.
        Use this in case you need to do something with all the outputs returned by
        `training_step`.

        :param outputs: List of outputs defined in 'training_step', or if there are
        multiple dataloaders, a list containing a list of outputs for each dataloader.
        """
        pass

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Conduct a validation step.

        Equivalent to `training_step` but with no optimization.
        Data is sampled by the validation dataloader.

        :param batch: Batch of validation data.
        :param batch_idx: Index of the batch.
        :return: Loss tensor
        """
        pass

    def validation_step_end(self, validation_step_outputs: List) -> Tensor:
        """
        Perform an action after a validation step completes.

        Mostly relevant for multi-GPU training as the `validation_step` method receives
        only a fraction (1/num_gpus) of the batch. If something should be done or
        calculated on the full batch, this gives the opportunity. The outputs of
        `validation_step` are collected in a list and are passed.
        """
        pass

    def validation_epoch_end(self, outputs: Any) -> None:
        """
        Perform an action after a validation epoch completes.

        Called at the end of the validation epoch with the outputs of all validation
        steps. Use this in case you need to do something with all the outputs returned
        by `training_step`.

        :param outputs: List of outputs defined in 'training_step', or if there are
        multiple dataloaders, a list containing a list of outputs for each dataloader.
        """
        pass

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """
        Conduct a test step.

        Equivalent to `training_step` but with no optimization.
        Data is sampled by the test dataloader.

        :param batch: Batch of test data.
        :param batch_idx: Index of the batch.
        :return: Loss tensor
        """
        pass

    def test_step_end(self, test_step_outputs: List) -> Tensor:
        """
        Perform an action after a validation step completes.

        Mostly relevant for multi-GPU training as the `test_step` method receives
        only a fraction (1/num_gpus) of the batch. If something should be done or
        calculated on the full batch, this gives the opportunity. The outputs of
        `test_step` are collected in a list and are passed.
        """
        pass

    def test_epoch_end(self, outputs: Any) -> None:
        """
        Perform an action after a test epoch completes.

        Called at the end of the test epoch with the outputs of all test steps.
        Use this in case you need to do something with all the outputs returned by
        `training_step`.

        :param outputs: List of outputs defined in 'training_step', or if there are
        multiple dataloaders, a list containing a list of outputs for each dataloader.
        """
        pass
