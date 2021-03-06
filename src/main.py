"""Main entrypoint for conducting training with PyTorch Lightning."""
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from data import LitDataModule
from module import LitModule
from utils.io import read_yml


def main() -> None:
    """Initialize and start a training run with PyTorch Lightning."""
    ###################################################
    # LOAD CONFIG FILE
    ###################################################
    cfg = read_yml('config.yml')

    ###################################################
    # INIT MODULE
    ###################################################
    module = LitModule(**cfg['MODULE'])

    ###################################################
    # INIT DATA MODULE
    ###################################################
    data_module = LitDataModule(**cfg['DATA'])

    ###################################################
    # INIT LOGGER
    ###################################################
    logger = TensorBoardLogger('logs')

    ###################################################
    # INIT TRAINER
    ###################################################
    trainer = Trainer(logger=logger, **cfg['TRAINER'])

    ###################################################
    # START TRAINING
    ###################################################
    trainer.fit(model=module, datamodule=data_module)


if __name__ == '__main__':
    main()
