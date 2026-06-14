from .training import Train
from typing import List
import torch
from components.lr_scheduler import LRScheduler
import logging
import torch.utils.data as utils


def training_factory(model: torch.nn.Module,
                     optimizers: List[torch.optim.Optimizer],
                     lr_schedulers: List[LRScheduler],
                     dataloaders: List[utils.DataLoader],
                     logger: logging.Logger) -> Train:

    return Train(model=model,
                optimizers=optimizers,
                lr_schedulers=lr_schedulers,
                dataloaders=dataloaders,
                logger=logger)

