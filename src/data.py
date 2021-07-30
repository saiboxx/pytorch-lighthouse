"""Contains data modules and datasets."""

from typing import Dict, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class LitDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        """Initialize a Lightning data module."""
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        """
        Prepare data.

        This involves tasks like e.g. downloading data, writing to disc, etc.
        In a distributed setting this is called on only 1 GPU.
        NOTE: DO NOT MAKE ANY ASSIGNMENTS HERE
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Perform data setup.

        Called at the beginning of fit (train + validate), validate, test, and predict.
        Called on every process in DDP.
        Make assignments here, e.g. loading the dataset to memory, data splits, etc.
        The `stage` parameter allows a different setup for every stage if desired.

        :param stage: String indicating the current stage ('fit', 'validate', 'test' or
        'predict').
        """
        pass

    def train_dataloader(self) -> DataLoader:
        """Return the training dataloader."""
        pass

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        pass

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        pass

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Clean up after stages.

        :param stage: String indicating the current stage ('fit', 'validate', 'test' or
        'predict').
        """
        pass


class LitDataSet(Dataset):
    """
    PyTorch Dataset.

    A PyTorch Dataset needs to at least implement the methods `__len__`
    and `__getitem__`.
    """

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict:
        """Return a sample from the dataset."""
        raise NotImplementedError
