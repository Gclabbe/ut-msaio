import argparse
import copy
from glob import glob
from os import path
from PIL import Image
import random

from typing import Dict, List, Optional, Union

from .config import WANDB_PROJECT

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from . import dense_transforms


# ripped from tests.py to calculate precision and recall on the fly
def point_in_box(pred, lbl):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return (x0 <= px) & (px < x1) & (y0 <= py) & (py < y1)


def point_close(pred, lbl, d=5):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return ((x0 + x1 - 1) / 2 - px) ** 2 + ((y0 + y1 - 1) / 2 - py) ** 2 < d**2


def box_iou(pred, lbl, t=0.5):
    px, py, pw2, ph2 = (
        pred[:, None, 0],
        pred[:, None, 1],
        pred[:, None, 2],
        pred[:, None, 3],
    )
    px0, px1, py0, py1 = px - pw2, px + pw2, py - ph2, py + ph2
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    iou = (
        abs(torch.min(px1, x1) - torch.max(px0, x0))
        * abs(torch.min(py1, y1) - torch.max(py0, y0))
    ) / (
        abs(torch.max(px1, x1) - torch.min(px0, x0))
        * abs(torch.max(py1, y1) - torch.min(py0, y0))
    )
    return iou > t


class PR:
    def __init__(self, min_size=20, is_close=point_in_box):
        self.min_size = min_size
        self.total_det = 0
        self.det = []
        self.is_close = is_close

    def add(self, d, lbl):
        lbl = torch.as_tensor(lbl.astype(float), dtype=torch.float32).view(-1, 4)
        d = torch.as_tensor(d, dtype=torch.float32).view(-1, 5)
        all_pair_is_close = self.is_close(d[:, 1:], lbl)

        # Get the box size and filter out small objects
        sz = abs(lbl[:, 2] - lbl[:, 0]) * abs(lbl[:, 3] - lbl[:, 1])

        # If we have detections find all true positives and count of the rest as false positives
        if len(d):
            detection_used = torch.zeros(len(d))
            # For all large objects
            for i in range(len(lbl)):
                if sz[i] >= self.min_size:
                    # Find a true positive
                    s, j = (
                        d[:, 0]
                        - 1e10 * detection_used
                        - 1e10 * ~all_pair_is_close[:, i]
                    ).max(dim=0)
                    if not detection_used[j] and all_pair_is_close[j, i]:
                        detection_used[j] = 1
                        self.det.append((float(s), 1))

            # Mark any detection with a close small ground truth as used (no not count false positives)
            detection_used += all_pair_is_close[:, sz < self.min_size].any(dim=1)

            # All other detections are false positives
            for s in d[detection_used == 0, 0]:
                self.det.append((float(s), 0))

        # Total number of detections, used to count false negatives
        self.total_det += int(torch.sum(sz >= self.min_size))

    @property
    def curve(self):
        true_pos, false_pos = 0, 0
        r = []
        for t, m in sorted(self.det, reverse=True):
            if m:
                true_pos += 1
            else:
                false_pos += 1
            prec = true_pos / (true_pos + false_pos)
            recall = true_pos / self.total_det
            r.append((prec, recall))
        return r

    @property
    def average_prec(self, n_samples=11):
        import numpy as np

        pr = np.array(self.curve, np.float32)
        return np.mean(
            [
                np.max(pr[pr[:, 1] >= t, 0], initial=0)
                for t in np.linspace(0, 1, n_samples)
            ]
        )


class DetectionSuperTuxDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        transform: Optional[dense_transforms.Compose] = None,
        min_size=20,
        sample_rate: Optional[float] = None,
    ):
        """
        Get all image info from the dense_data directory for retrieval later

        :param dataset_path: the path to the dense_data directory
        :type dataset_path: str
        :param transform: the transform to apply to the images
        :type transform: Optional[dense_transforms.Compose]
        :param min_size: the minimum size of a box to be considered
        :type min_size: int
        :param sample_rate: the rate at which to sample the dataset
        :type sample_rate: Optional[float]
        """
        self.dataset_path = dataset_path
        self.transform = transform or dense_transforms.ToTensor()

        self.files = []
        for im_f in glob(path.join(dataset_path, "*_im.jpg")):
            self.files.append(im_f.replace("_im.jpg", ""))

        if sample_rate is not None:
            self.files = random.sample(self.files, int(len(self.files) * sample_rate))

        self.images = []
        self.boxes = []
        for im_f in self.files:
            with Image.open(f"{im_f}_im.jpg") as f:
                self.images.append(copy.deepcopy(f))  # self.transform(img)

            # with open(f"{im_f}_boxes.npz", 'rb') as f:
            #     self.boxes.append(np.load(f))

        self.min_size = min_size

    def _filter(self, boxes):
        """
        filter out boxes that are too small

        :param boxes: the boxes to filter
        :type boxes: np.ndarray
        :return: the filtered boxes
        :rtype: np.ndarray
        """
        if len(boxes) == 0:
            return boxes
        return boxes[
            abs(boxes[:, 3] - boxes[:, 1]) * abs(boxes[:, 2] - boxes[:, 0])
            >= self.min_size
        ]

    def __len__(self):
        """
        return the number of images in the dataset

        :return: the number of images in the dataset
        :rtype: int
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        return a tuple: img, label

        :param idx: index of the image to return
        :type idx: int
        :return: tuple of the image and its labels
        :rtype: (Image.Image, Image.Image)
        """
        b = self.files[idx]
        image = self.images[idx]
        # im = Image.open(b + "_im.jpg")
        boxes = np.load(b + "_boxes.npz")
        # boxes = self.boxes[idx]
        data = (
            image,
            self._filter(boxes["karts"]),
            self._filter(boxes["bombs"]),
            self._filter(boxes["pickup"]),
        )
        if self.transform is not None:
            data = self.transform(*data)

        return data


def load_detection_data(
    dataset_path: str,
    num_workers: int = 0,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: Optional[int] = None,
    sample_rate: Optional[float] = None,
    # transform: Optional[dense_transforms.Compose] = None,
    **kwargs,
) -> DataLoader:
    """
    Load the detection data from the dataset path

    :param dataset_path: the path to the dataset
    :type dataset_path: str
    :param num_workers: the number of workers to use
    :type num_workers: int
    :param batch_size: the batch size
    :type batch_size: int
    :param seed: the random seed
    :type seed: Optional[int]
    :param sample_rate: the rate at which to sample the dataset
    :type sample_rate: Optional[float]
    :param shuffle: whether to shuffle the dataset, defaults to True
    :type shuffle: bool,
    :return: the DataLoader
    :rtype: torch.DataLoader
    """
    if seed is not None:
        torch.manual_seed(seed)

    return DataLoader(
        DetectionSuperTuxDataset(
            dataset_path=dataset_path, sample_rate=sample_rate, **kwargs
        ),
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        **kwargs,
    )


def compute_params(model: torch.nn.Module) -> int:
    """
    Compute the number of trainable parameters in the model

    :param model: the model to compute the parameters for
    :type model: torch.nn.Module
    :return: the number of trainable parameters
    :rtype: int
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def init_wandb(
    args: argparse.Namespace,
    run_id: str,
    params: int,
    fold: Optional[int] = None,
    class_distribution: Optional[List] = None,
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
    class_distribution = class_distribution or [1] * args.num_classes

    # if the import failed, don't do anything
    if wandb is None:
        return

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
            "first_step": args.first_step,
            "params": params,
            "class_distro": [round(x, 4) for x in class_distribution],
        },
        tags=wandb_tags,
    )


if __name__ == "__main__":
    import torchvision.transforms.functional as F
    from pylab import show, subplots
    import matplotlib.patches as patches
    import numpy as np

    # show non-transformed images with target boxes
    dataset = DetectionSuperTuxDataset("dense_data/train")  # , sample_rate=0.1)

    fig, axs = subplots(1, 2)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[100 + i]
        ax.imshow(F.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle(
                    (k[0] - 0.5, k[1] - 0.5),
                    width=k[2] - k[0],
                    height=k[3] - k[1],
                    fc="none",
                    ec="r",
                    lw=2,
                )
            )
        for k in bomb:
            ax.add_patch(
                patches.Rectangle(
                    (k[0] - 0.5, k[1] - 0.5),
                    k[2] - k[0],
                    k[3] - k[1],
                    fc="none",
                    ec="g",
                    lw=2,
                )
            )
        for k in pickup:
            ax.add_patch(
                patches.Rectangle(
                    (k[0] - 0.5, k[1] - 0.5),
                    k[2] - k[0],
                    k[3] - k[1],
                    fc="none",
                    ec="b",
                    lw=2,
                )
            )
        ax.axis("off")
    fig.tight_layout()
    # fig.savefig('box.png', bbox_inches='tight', pad_inches=0, transparent=True)

    # show transformed images with detection heatmaps
    # dataset = DetectionSuperTuxDataset(
    #     "dense_data/train",
    #     transform=dense_transforms.Compose(
    #         [dense_transforms.RandomHorizontalFlip(0), dense_transforms.ToTensor()]
    #     ),
    # )
    #
    # fig, axs = subplots(1, 2)
    # for i, ax in enumerate(axs.flat):
    #     im, *dets = dataset[100 + i]
    #     hm, size = dense_transforms.detections_to_heatmap(dets, im.shape[1:])
    #     ax.imshow(F.to_pil_image(im), interpolation=None)
    #     hm = hm.numpy().transpose([1, 2, 0])
    #     alpha = 0.25 * hm.max(axis=2) + 0.75
    #     r = 1 - np.maximum(hm[:, :, 1], hm[:, :, 2])
    #     g = 1 - np.maximum(hm[:, :, 0], hm[:, :, 2])
    #     b = 1 - np.maximum(hm[:, :, 0], hm[:, :, 1])
    #     ax.imshow(np.stack((r, g, b, alpha), axis=2), interpolation=None)
    #     ax.axis("off")
    # fig.tight_layout()
    # fig.savefig('heat.png', bbox_inches='tight', pad_inches=0, transparent=True)

    show()
