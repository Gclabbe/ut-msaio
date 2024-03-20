import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from . import dense_transforms

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(
        (
            (
                weights.sum(1)
                * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]
            ).sum(1),
            (
                weights.sum(2)
                * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]
            ).sum(1),
        ),
        1,
    )

class Planner(torch.nn.Module):
    # Planner stripped


def save_model(model):
    from torch import save
    from os import path

    if isinstance(model, Planner):
        return save(
            model.state_dict(),
            path.join(path.dirname(path.abspath(__file__)), "planner.th"),
        )
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path

    r = Planner()
    r.load_state_dict(
        load(
            path.join(path.dirname(path.abspath(__file__)), "planner.th"),
            map_location="cpu",
        )
    )
    return r


if __name__ == "__main__":
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Testing on {device.type}")

    def test_planner(args):
        # Load model
        planner = load_model().to(device).eval()
        pytux = PyTux()
        for t in args.track:
            scores = []
            for i in range(args.retries):
                steps, how_far = pytux.rollout(
                    t,
                    control,
                    planner=planner,
                    max_frames=args.n_frames,
                    verbose=args.verbose,
                )
                score = 10 if steps < (args.n_frames - 1) else round(how_far * 10, 0)
                scores.append(int(score))
                average_score = np.average(scores)
                print(f"{steps}: {how_far :0.3f} --> {np.sum(scores) :d} --> {average_score :0.2f}")
            print(
                f"{i}/{args.retries}) Score for track {t}: {np.average(scores) :0.2f}"
            )
            unique, counts = np.unique(scores, return_counts=True)
            print([f"{x :>4d}" for x in unique])
            print([f"{x :>4d}" for x in counts])
            print([f"{int(100 * x / args.retries) :>3d}%" for x in counts])
        pytux.close()

    parser = ArgumentParser("Test the planner")
    parser.add_argument("track", nargs="+")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-n", "--n_frames", default=1000, type=int)
    parser.add_argument("--retries", default=1, type=int)
    args = parser.parse_args()
    test_planner(args)
