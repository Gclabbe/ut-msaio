from .grader import Grader, Case, MultiCase

import numpy as np
import torchvision.transforms.functional as TF
from tqdm import trange
from os import path
import torch
import pystk

RESCUE_TIMEOUT = 30
TRACK_OFFSET = 15

pystk_config = pystk.GraphicsConfig.hd()
pystk_config.screen_width = 128
pystk_config.screen_height = 96

pystk.init(pystk_config)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Running on {device.type}")


class PySTKGrader(Grader):
    use_planner = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.C = self.module.control
        self.P = None
        if self.use_planner:
            self.P = self.module.load_model().to(device).eval()

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
        return np.array([p[0] / p[-1], -p[1] / p[-1]])

    def _test(self, track, max_frames):
        config = pystk.RaceConfig(num_kart=1, laps=1)
        config.track = track
        config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL

        k = pystk.Race(config)
        try:
            state = pystk.WorldState()
            track = pystk.Track()

            k.start()
            k.step()

            last_rescue = 0
            for t in trange(max_frames):
                state.update()
                track.update()

                kart = state.players[0].kart

                if kart.race_result:
                    break

                if self.P is None:
                    proj = np.array(state.players[0].camera.projection).T
                    view = np.array(state.players[0].camera.view).T
                    aim_point_world = self._point_on_track(
                        kart.distance_down_track + TRACK_OFFSET, track
                    )
                    aim_point_image = self._to_image(aim_point_world, proj, view)
                else:
                    # image = image.to(device)
                    image = np.array(k.render_data[0].image)
                    image = TF.to_tensor(image).to(device)
                    aim_point_image = (
                        self.P(image[None]).squeeze(0).cpu().detach().numpy()
                    )

                current_vel = np.linalg.norm(kart.velocity)
                action = self.C(aim_point_image, current_vel)

                if current_vel <= 1.0 and t - last_rescue > RESCUE_TIMEOUT:
                    action.rescue = True
                    last_rescue = t

                k.step(action)
        finally:
            k.stop()
            del k
        if kart.race_result:
            return 1, "%0.1f s" % kart.finish_time
        return kart.overall_distance / track.length, "%0.1f%% done" % (
            100 * kart.overall_distance / track.length
        )


class ControllerGrader(PySTKGrader, Grader):
    """Controller"""

    use_planner = False

    @Case(score=5)
    def test_lighthouse(self):
        """lighthouse"""
        return self._test("lighthouse", 550)

    @Case(score=5)
    def test_hacienda(self):
        """hacienda"""
        return self._test("hacienda", 700)

    @Case(score=5)
    def test_snowtuxpeak(self):
        """snowtuxpeak"""
        return self._test("snowtuxpeak", 700)

    @Case(score=5)
    def test_zengarden(self):
        """zengarden"""
        return self._test("zengarden", 600)

    @Case(score=5)
    def test_cornfield_crossing(self):
        """cornfield_crossing"""
        return self._test("cornfield_crossing", 750)

    @Case(score=5)
    def test_scotland(self):
        """scotland"""
        return self._test("scotland", 700)


class PlannerGrader(PySTKGrader, Grader):
    """Planner"""

    use_planner = True

    @Case(score=10)
    def test_lighthouse(self, it=0):
        """lighthouse"""
        return self._test("lighthouse", 650)

    @Case(score=10)
    def test_hacienda(self, it=0):
        """hacienda"""
        return self._test("hacienda", 700)

    @Case(score=10)
    def test_snowtuxpeak(self, it=0):
        """snowtuxpeak"""
        return self._test("snowtuxpeak", 700)

    @Case(score=10)
    def test_zengarden(self, it=0):
        """zengarden"""
        return self._test("zengarden", 700)

    @Case(score=10)
    def test_cornfield_crossing(self, it=0):
        """cornfield_crossing"""
        return self._test("cornfield_crossing", 950)

    @Case(score=10)
    def test_scotland(self, it=0):
        """scotland"""
        return self._test("scotland", 850)


class NewLevelrGrader(PySTKGrader, Grader):
    """Test level"""

    use_planner = True

    @Case(score=10)
    def test_cocoa_temple(self, it=0):
        """cocoa temple"""
        return self._test("cocoa_temple", 800)
