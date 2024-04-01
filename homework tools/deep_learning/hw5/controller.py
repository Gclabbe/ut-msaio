from argparse import ArgumentParser

import numpy as np
import pystk

from .utils import PyTux


def control(
    aim_point,
    current_vel,
    target_velocity,
):
    """
    Set the Action for the low-level controller

    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """
    raise NotImplementedError("control")


if __name__ == "__main__":

    def test_controller(args):
        pytux = PyTux()
        for t in args.track:
            for _ in range(args.retries):
                steps, how_far = pytux.rollout(
                    t, control, max_frames=args.n_images, verbose=args.verbose
                )
                print(steps, how_far)
        pytux.close()

    parser = ArgumentParser()
    parser.add_argument("track", nargs="+")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-d" "--debug", action="store_true")
    parser.add_argument("-t", "--target_velocity", default=5.0, type=float)
    parser.add_argument("-n", "--n_images", default=100, type=int)
    parser.add_argument("--retries", default=1, type=int)

    args = parser.parse_args()
    test_controller(args)
