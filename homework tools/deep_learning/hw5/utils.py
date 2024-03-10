import numpy as np
import random

# ToDo: remove for the POS grading system
try:
    from tqdm import trange
except ImportError:
    trange = range

from typing import Optional

import pystk
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from . import dense_transforms

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15
DATASET_PATH = "drive_data"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Training on {device.type}")


class SuperTuxDataset(Dataset):
    def __init__(
        self,
        dataset_path=DATASET_PATH,
        sample_rate: Optional[float] = None,
        transform=dense_transforms.ToTensor(),
    ):
        from PIL import Image
        from glob import glob
        from os import path

        self.data = []
        if sample_rate is not None:
            self.files = random.sample(self.files, int(len(self.files) * sample_rate))

        for f in glob(path.join(dataset_path, "*.csv")):
            i = Image.open(f.replace(".csv", ".png"))
            i.load()
            self.data.append((i, np.loadtxt(f, dtype=np.float32, delimiter=",")))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = self.transform(*data)
        return data


def load_data(
    dataset_path=DATASET_PATH,
    transform=dense_transforms.ToTensor(),
    num_workers=0,
    batch_size=128,
    sample_rate: Optional[float] = None,
):
    dataset = SuperTuxDataset(
        dataset_path,
        sample_rate=sample_rate,
        transform=transform,
    )
    return DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )


class PyTux:
    _singleton = None

    def __init__(self, screen_width=128, screen_height=96):
        assert PyTux._singleton is None, "Cannot create more than one pytux object"
        PyTux._singleton = self
        self.config = pystk.GraphicsConfig.hd()
        self.config.screen_width = screen_width
        self.config.screen_height = screen_height
        pystk.init(self.config)
        self.k = None

    @staticmethod
    def _point_on_track(distance, track, offset=0.0):
        """
        Get a point at `distance` down the `track`. Optionally applies an offset after the track segment if found.
        Returns a 3d coordinate
        """
        node_idx = np.searchsorted(
            track.path_distance[..., 1], distance % track.path_distance[-1, 1]
        ) % len(track.path_nodes)
        d = track.path_distance[node_idx]
        x = track.path_nodes[node_idx]
        t = (distance + offset - d[0]) / (d[1] - d[0])
        return x[1] * t + x[0] * (1 - t)

    @staticmethod
    def _to_image(x, proj, view):
        p = proj @ view @ np.array(list(x) + [1])
        return np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1)

    def rollout(
        self,
        track,
        controller,
        planner=None,
        max_frames=1000,
        verbose=False,
        data_callback=None,
    ):
        """
        Play a level (track) for a single round.
        :param track: Name of the track
        :param controller: low-level controller, see controller.py
        :param planner: high-level planner, see planner.py
        :param max_frames: Maximum number of frames to play for
        :param verbose: Should we use matplotlib to show the agent drive?
        :param data_callback: Rollout calls data_callback(time_step, image, 2d_aim_point) every step, used to store the
                              data
        :return: Number of steps played
        """
        if self.k is not None and self.k.config.track == track:
            self.k.restart()
            self.k.step()
        else:
            if self.k is not None:
                self.k.stop()
                del self.k
            config = pystk.RaceConfig(num_kart=1, laps=1, track=track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

            self.k = pystk.Race(config)
            self.k.start()
            self.k.step()

        state = pystk.WorldState()
        track = pystk.Track()

        last_rescue = 0

        if verbose:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1)

        for t in trange(max_frames):
            state.update()
            track.update()

            kart = state.players[0].kart

            if np.isclose(kart.overall_distance / track.length, 1.0, atol=2e-3):
                if verbose:
                    print("Finished at t=%d" % t)
                break

            proj = np.array(state.players[0].camera.projection).T
            view = np.array(state.players[0].camera.view).T

            aim_point_world = self._point_on_track(
                kart.distance_down_track + TRACK_OFFSET, track
            )
            aim_point_image = self._to_image(aim_point_world, proj, view)
            if data_callback is not None:
                data_callback(t, np.array(self.k.render_data[0].image), aim_point_image)

            if planner:
                planner = planner.to(device).eval()
                image = np.array(self.k.render_data[0].image)
                image = TF.to_tensor(image).to(device)
                aim_point_image = planner(image[None]).squeeze(0).cpu().detach().numpy()

            current_vel = np.linalg.norm(kart.velocity)
            action = controller(aim_point_image, current_vel)

            if current_vel < 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                last_rescue = t
                action.rescue = True

            if verbose:
                ax.clear()
                ax.imshow(self.k.render_data[0].image)
                WH2 = (
                    np.array([self.config.screen_width, self.config.screen_height]) / 2
                )
                ax.add_artist(
                    plt.Circle(
                        WH2 * (1 + self._to_image(kart.location, proj, view)),
                        2,
                        ec="b",
                        fill=False,
                        lw=1.5,
                    )
                )
                ax.add_artist(
                    plt.Circle(
                        WH2 * (1 + self._to_image(aim_point_world, proj, view)),
                        2,
                        ec="r",
                        fill=False,
                        lw=1.5,
                    )
                )
                if planner:
                    ap = self._point_on_track(
                        kart.distance_down_track + TRACK_OFFSET, track
                    )
                    ax.add_artist(
                        plt.Circle(
                            WH2 * (1 + aim_point_image), 2, ec="g", fill=False, lw=1.5
                        )
                    )
                plt.pause(1e-3)

            self.k.step(action)
            t += 1
        return t, kart.overall_distance / track.length

    def close(self):
        """
        Call this function, once you're done with PyTux
        """
        if self.k is not None:
            self.k.stop()
            del self.k
        pystk.clean()


if __name__ == "__main__":
    from .controller import control
    from argparse import ArgumentParser
    from os import makedirs

    def noisy_control(aim_pt, vel):
        return control(
            aim_pt + np.random.randn(*aim_pt.shape) * aim_noise,
            vel + np.random.randn() * vel_noise,
        )

    parser = ArgumentParser("Collects a dataset for the high-level planner")
    parser.add_argument("track", nargs="+")
    parser.add_argument("-o", "--output", default=DATASET_PATH)
    parser.add_argument("-n", "--n_images", default=10000, type=int)
    parser.add_argument("-m", "--steps_per_track", default=20000, type=int)
    parser.add_argument("--aim_noise", default=0.1, type=float)
    parser.add_argument("--vel_noise", default=5, type=float)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    try:
        makedirs(args.output)
    except OSError:
        pass
    pytux = PyTux()
    for track in args.track:
        n, images_per_track = 0, args.n_images // len(args.track)
        aim_noise, vel_noise = 0, 0

        def collect(_, im, pt):
            from PIL import Image
            from os import path

            global n
            id = n if n < images_per_track else np.random.randint(0, n + 1)
            if id < images_per_track:
                fn = path.join(args.output, track + "_%05d" % id)
                Image.fromarray(im).save(fn + ".png")
                with open(fn + ".csv", "w") as f:
                    f.write("%0.1f,%0.1f" % tuple(pt))
            n += 1

        while n < args.steps_per_track:
            steps, how_far = pytux.rollout(
                track,
                noisy_control,
                max_frames=1000,
                verbose=args.verbose,
                data_callback=collect,
            )
            print(steps, how_far)
            # Add noise after the first round
            aim_noise, vel_noise = args.aim_noise, args.vel_noise
    pytux.close()
