import argparse
from os import getcwd, makedirs, path
from typing import Dict, List, Optional, Tuple, Union

# import matplotlib.pyplot.axes as axes
import matplotlib.patches as patches
import torch
from torch import load
from torch import save
import torchvision.transforms.functional as TF

from pylab import show, subplots

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_peak(
):
    """
    Extract local maxima (peaks) in a 2d heatmap.

    Returns the list of peaks as (score, cx, cy) where cx, cy are the position of a
    peak and score is the heatmap value at the peak.

    """
    raise NotImplementedError("extract_peak")


def extract_height_width(
):
    """
    Extract boundaries around the peak of a heatmap.  This routine is not used.
    It was created when I was trying to figure out how to extract the width and height of the bounding box.

    """
    raise NotImplementedError("extract_height_width")


class Detector(torch.nn.Module):
    def __init__(
        self,
    ):
        """
        Detector model that uses a FCN to detect peaks of objects in an image as well
        as the potential H/W of the bounding box surrounding the object.

        """
        super(Detector, self).__init__()
        raise NotImplementedError("Detector.__init__")

    def forward(self, x):
        raise NotImplementedError("Detector.forward")

    def detect(self, image):
        """
        Implement object detection here.
        @image: 3 x H x W image
        @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                 return no more than 30 detections per image per class. You only need to predict width and height
                 for extra credit. If you do not predict an object size, return w=0, h=0.
        Hint: Use extract_peak here
        Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
              scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
              out of memory.

        """
        raise NotImplementedError("Detector.detect")


def save_model(
    model,
    base_path: str = "models",
    save_checkpoint: Optional[bool] = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
):
    """
    Save the model to a file

    Removed all of my modified code except for an example of how to save
    a checkpoint instead of just the model to restore optimizer and scheduler state
    """

    filename = f"{base_path}\\det.th"
    print(f"!!! Saving {filename}")
    save(model.state_dict(), filename)

    if save_checkpoint:
        print(f"!!! Saving checkpoint {base_path}\\checkpoint.pth")
        state = {
            "epoch": epoch + 1,
            "loss_train": loss_train,
            "loss_valid": loss_valid if loss_valid is not None else 0,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        save(state, f"{base_path}\\checkpoint.pth")

    return True


def load_model(model: Optional[Detector] = None) -> Detector:
    """
    Provided code: Load the model from a file

    :param model: The model to load, defaults to None
    :type model: Detector, optional
    :return: The loaded model
    :rtype: Detector
    """
    model = model or Detector()
    model.load_state_dict(
        load(
            path.join(path.dirname(path.abspath(__file__)), "det.th"),
            map_location="cpu",
        )
    )
    return model


def create_patch(object: Tuple[int] , color: str) -> patches.Rectangle:
    """
    Provided code: Create a patch for an object to display in matplotlib

    :param object: The object to create a patch for
    :type object: Tuple[int]
    :param color: The color for the patch
    :type color: str
    """
    return patches.Rectangle(
        (object[0] - 0.5, object[1] - 0.5),
        object[2] - object[0],
        object[3] - object[1],
        facecolor="none",
        edgecolor=color,
    )


def draw_boxes(ax, kart: List[Tuple[int]], bomb: List[Tuple[int]], pickup: List[Tuple[int]]):
    """
    Provided code: Draw the boxes for the objects in the image
    Bit odd to just repeat the same code for each object type, but it's provided code

    :param ax: The axis to draw the boxes on
    :type ax: matplotlib.axes
    :param kart: The kart objects
    :type kart: List[Tuple[int]]
    :param bomb: The bomb objects
    :type bomb: List[Tuple[int]]
    :param pickup: The pickup objects
    :type pickup: List[Tuple[int]]
    :return: The axis with the boxes drawn
    :rtype: matplotlib.axes
    """
    for k in kart:
        ax.add_patch(create_patch(k, "r"))
        w, h = k[2] - k[0], k[3] - k[1]
        ax.text(k[0], k[1], f"{w:.2f},{h:.2f}", color="yellow", fontsize=8)
    for b in bomb:
        ax.add_patch(create_patch(b, "g"))
        w, h = b[2] - b[0], b[3] - b[1]
        ax.text(b[0], b[1], f"{w:.2f},{h:.2f}", color="yellow", fontsize=8)
    for p in pickup:
        w, h = p[2] - p[0], p[3] - p[1]
        ax.text(p[0], p[1], f"{w: .2f},{h: .2f}", color="yellow", fontsize=8)
        ax.add_patch(create_patch(p, "b"))

    return ax


def draw_detections(ax, detections: List[List[Tuple[float, int, int, float, float]]], radius_min: float = 0.1):
    """
    Provided code: Draw the detections for the objects in the image
    Will break if we try to run this with more or less than 3 object types

    :param ax: The axis to draw the detections on
    :type ax: matplotlib.axes
    :param detections: The detections for the objects
    :type detections: List[List[Tuple[float, int, int, float, float]]]
    :param radius_min: The minimum radius for the detection, defaults to 0.1
    :type radius_min: float, optional
    :return: The axis with the detections drawn
    :rtype: matplotlib.axes
    """
    for c in range(2, 3):
        for s, cx, cy, w, h in detections[c]:
            radius = max(2 + s / 2, radius_min)
            ax.add_patch(patches.Circle(xy=(cx, cy), radius=radius, color="rgb"[c]))
            ax.text(
                cx - radius / 2,
                cy - radius / 2,
                f"{w * 2 :.2f},{h * 2 :.2f}",
                color="white",
                fontsize=8,
            )
            ax.add_patch(
                patches.Rectangle(
                    xy=(cx - w, cy - h),
                    width=w * 2,
                    height=h * 2,
                    facecolor="none",
                    edgecolor="rgb"[c],
                )
            )

    return ax


if __name__ == "__main__":
    from .utils import DetectionSuperTuxDataset

    """
    Shows detections of your detector
    Modified by myself to als show the H/W for ground truth (yellow) and detections (white)
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir")
    # Put custom arguments here
    parser.add_argument("-n", "--num_images", type=int, default=12)
    parser.add_argument("-dr", "--min_det_radius", type=float, default=0.1)
    parser.add_argument("-t", "--transform", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device", device)

    r = Detector().to(device)
    model = load_model().eval().to(device)

    dataset = DetectionSuperTuxDataset(
        "dense_data/valid", sample_rate=0.0008, min_size=0
    )

    # modify this to show more images.  Right now it's set to 2
    # Tried briefly to show 1, didn't care enough to figure out how to make that work
    fig, axs = subplots(2, 2)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        ax = draw_boxes(ax, kart, bomb, pickup)
        detections = model.detect(im.to(device))
        ax = draw_detections(ax, detections)
        ax.axis("off")
        # break
    show()
