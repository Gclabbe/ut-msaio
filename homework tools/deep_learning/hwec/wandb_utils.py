import argparse
from typing import List, Optional

from .config import WANDB_PROJECT
try:
    import wandb
except ImportError:
    wandb = None


def init_wandb(
    args: argparse.Namespace,
    run_id: str,
    params: int,
    fold: Optional[int] = None,
    # class_distribution: Optional[List] = None,
    wandb_tags: Optional[List] = None,
):
    """
    Initialize Weights & Biases to log the training process

    :param args: the arguments from argparse
    :type args: argparse.Namespace
    :param run_id: the run id for wandb
    :type run_id: str
    :param params: the number of parameters in the model
    :type params: int
    :param fold: the fold number for cross-validation
    :type fold: Optional[int]
    :param class_distribution: the class distribution for the loss function
    :type class_distribution: Optional[List]
    :param wandb_tags: the tags for wandb
    :type wandb_tags: Optional[List]
    """
    # if the import failed, don't do anything
    if wandb is None:
        return False

    wandb_tags = wandb_tags or []

    wandb.init(
        project=WANDB_PROJECT[args.model],
        name=f"{run_id}_fold_{fold}" if fold is not None else run_id,
        config={
            "epochs": args.epochs,
            "train_batch_size": args.train_batch_size,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "dropout_rate": args.dropout_rate,
            "depth": args.depth,
            # "first_step": args.first_step,
            "params": params,
            # "class_distro": [round(x, 4) for x in class_distribution],
        },
        tags=wandb_tags,
    )
