"""
EDIT THIS FILE AT YOUR OWN RISK!
It will not ship with your code, editing it will only change the test cases locally, and might make you fail our
remote tests.
"""

import torch
from .grader import Grader, Case

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, *args, **kwargs):
        return x


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def point_in_box(pred, lbl):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return (x0 <= px) & (px < x1) & (y0 <= py) & (py < y1)


def point_close(pred, lbl, d=5):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return ((x0 + x1 - 1) / 2 - px) ** 2 + ((y0 + y1 - 1) / 2 - py) ** 2 < d**2


def box_iou(pred, lbl, t=0.5):
    # print("pred", pred[0])
    # print("lbl", lbl[0])
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


class ExtractPeakGrader(Grader):
    """extract_peak"""

    def test_det(self, p, hm, min_score=0):
        centers = [(cx, cy) for s, cx, cy in p]
        assert len(centers) == len(set(centers)), "Duplicate detection"
        assert all(
            [0 <= cx < hm.size(1) and 0 <= cy < hm.size(0) for cx, cy in centers]
        ), "Peak out of bounds"
        assert all(
            [s > min_score for s, cx, cy in p]
        ), "Returned a peak below min_score"
        assert all([s == hm[cy, cx] for s, cx, cy in p]), "Score does not match heatmap"

    @Case(score=5)
    def test_format(self, min_score=0):
        """return value"""
        ep = self.module.extract_peak
        for i in range(50, 200, 10):
            img = torch.randn(3 * i, 2 * i)
            p = ep(img, max_pool_ks=3, min_score=min_score, max_det=i)
            assert len(p) <= i, "Expected at most %d peaks, got %d" % (i, len(p))
            self.test_det(p, img, min_score=min_score)

    @Case(score=5)
    def test_radius1(self, min_score=0):
        """radius=1"""
        img = torch.randn(54, 123)
        p = self.module.extract_peak(
            img, max_pool_ks=1, min_score=min_score, max_det=100000
        )
        assert len(p) == (img > 0).sum(), "Expected exactly %d detections, got %d" % (
            len(p),
            (img > 0).sum(),
        )
        self.test_det(p, img, min_score=min_score)

        """negative min_score test"""
        p = self.module.extract_peak(img, max_pool_ks=1, min_score=-1e5, max_det=100000)
        assert (
            len(p) == (img > -1e5).sum()
        ), "Expected exactly %d detections, got %d" % ((img > -1e5).sum(), len(p))
        self.test_det(p, img, min_score=-1e5)

        import numpy as np

        min_s = img.min()
        max_s = img.max()
        for _min_score in np.linspace(min_s, max_s, 5):
            """min_score tests"""
            p = self.module.extract_peak(
                img, max_pool_ks=1, min_score=_min_score, max_det=100000
            )
            assert (
                len(p) == (img > _min_score).sum()
            ), "Expected exactly %d detections, got %d" % (
                (img > _min_score).sum(),
                len(p),
            )
            self.test_det(p, img, min_score=_min_score)

    @Case(score=5)
    def test_manyl(self, min_score=0, max_pool_ks=5):
        """peak extraction"""
        from functools import partial

        ep = partial(
            self.module.extract_peak,
            max_pool_ks=max_pool_ks,
            min_score=min_score,
            max_det=100,
        )
        assert len(ep(torch.zeros((10, 10)))) == 0, "No peak expected"
        assert (
            len(ep(torch.arange(100).view(10, 10).float())) == 1
        ), "Single peak expected"
        assert len(ep(torch.ones((10, 10)))) == 100, "100 peaks expected"
        assert (
            len(ep((torch.arange(100).view(10, 10) == 55).float())) == 1
        ), "Single peak expected"
        assert (
            len(ep((torch.arange(100).view(10, 10) == 55).float() - 1)) == 0
        ), "No peak expected"

    @Case(score=5)
    def test_random(self, min_score=0, max_pool_ks=5):
        """randomized test"""
        from functools import partial

        ep = partial(
            self.module.extract_peak,
            max_pool_ks=max_pool_ks,
            min_score=min_score,
            max_det=100,
        )
        img = torch.zeros((100, 100))
        c = torch.randint(0, 100, (100, 2))
        pts = set()
        for i, p in enumerate(c):
            if i == 0 or (c[:i] - p[None]).abs().sum(dim=1).min() > max_pool_ks:
                pts.add((float(p[0]), float(p[1])))
                img[p[1], p[0]] = 1
                if len(pts) >= 10:
                    break
        p_img = 1 * img
        for k in range(1, max_pool_ks + 1, 2):
            p_img += torch.nn.functional.avg_pool2d(
                img[None, None], k, padding=k // 2, stride=1
            )[0, 0]
            p = ep(p_img)
            self.test_det(p, p_img, min_score)
            ret_pts = {(float(cx), float(cy)) for s, cx, cy in p}
            assert ret_pts == pts, "Returned the wrong peaks for randomized test"


class DetectorGrader(Grader):
    """Detector"""

    @Case(score=5)
    def test_format(self):
        """return value"""
        det = self.module.load_model().eval()

        # ToDo: remove sampling when really testing
        dataset = self.module.utils.DetectionSuperTuxDataset(
            "dense_data/valid", min_size=0
        )  # , sample_rate=0.01)

        for i, (img, *gts) in enumerate(tqdm(dataset, desc="Detector Grading")):
            d = det.detect(img)
            assert len(d) == 3, "Return three lists of detections"
            assert (
                len(d[0]) <= 30 and len(d[1]) <= 30 and len(d[2]) <= 30
            ), "Returned more than 30 detections per class"
            assert all(
                len(i) == 5 for c in d for i in c
            ), "Each detection should be a tuple (score, cx, cy, w/2, h/2)"
            if i > 10:
                break


class DetectionGrader(Grader):
    """Detection model"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        det = self.module.load_model().eval().to(device)

        # Compute detections
        self.pr_box = [PR() for _ in range(3)]
        self.pr_dist = [PR(is_close=point_close) for _ in range(3)]
        self.pr_iou = [PR(is_close=box_iou) for _ in range(3)]

        dataset = self.module.utils.DetectionSuperTuxDataset(
            "dense_data/valid",
            min_size=0,
            # ToDo: remove sampling when really testing
            # sample_rate=0.001,
        )

        for img, *gts in tqdm(dataset, desc="Detection Grading"):
            with torch.no_grad():
                detections = det.detect(img.to(device))
                # print(gts)
                for i, gt in enumerate(gts):
                    self.pr_box[i].add(detections[i], gt)
                    self.pr_dist[i].add(detections[i], gt)
                    self.pr_iou[i].add(detections[i], gt)

    @staticmethod
    def compute_score(ap, min_val, max_val):
        return max(min(ap, max_val) - min_val, 0) / (max_val - min_val)

    @staticmethod
    def return_string(ap, score, max_score, min_val, max_val):
        return f"AP = {ap :>5.3f}: {score * max_score :>4.1f}/{max_score} (min {min_val :<4.2f}, max {max_val :<4.2f})"

    @Case(score=10)
    def test_box_ap0(self, min_val=0.5, max_val=0.75):
        """Average precision (inside box c=0)"""
        ap = self.pr_box[0].average_prec
        score = self.compute_score(ap=ap, min_val=min_val, max_val=max_val)
        return_str = self.return_string(
            ap=ap, score=score, max_score=10, min_val=min_val, max_val=max_val
        )
        return score, return_str

    @Case(score=10)
    def test_box_ap1(self, min_val=0.25, max_val=0.45):
        """Average precision (inside box c=1)"""
        ap = self.pr_box[1].average_prec
        score = self.compute_score(ap=ap, min_val=min_val, max_val=max_val)
        return_str = self.return_string(
            ap=ap, score=score, max_score=10, min_val=min_val, max_val=max_val
        )
        return score, return_str

    @Case(score=10)
    def test_box_ap2(self, min_val=0.6, max_val=0.85):
        """Average precision (inside box c=2)"""
        ap = self.pr_box[2].average_prec
        score = self.compute_score(ap=ap, min_val=min_val, max_val=max_val)
        return_str = self.return_string(
            ap=ap, score=score, max_score=10, min_val=min_val, max_val=max_val
        )
        return score, return_str

    @Case(score=15)
    def test_dist_ap0(self, min_val=0.5, max_val=0.72):
        """Average precision (distance c=0)"""
        ap = self.pr_dist[0].average_prec
        score = self.compute_score(ap=ap, min_val=min_val, max_val=max_val)
        return_str = self.return_string(
            ap=ap, score=score, max_score=15, min_val=min_val, max_val=max_val
        )
        return score, return_str

    @Case(score=15)
    def test_dist_ap1(self, min_val=0.25, max_val=0.45):
        """Average precision (distance c=1)"""
        ap = self.pr_dist[1].average_prec
        score = self.compute_score(ap=ap, min_val=min_val, max_val=max_val)
        return_str = self.return_string(
            ap=ap, score=score, max_score=15, min_val=min_val, max_val=max_val
        )
        return score, return_str

    @Case(score=15)
    def test_dist_ap2(self, min_val=0.6, max_val=0.85):
        """Average precision (distance c=2)"""
        ap = self.pr_dist[2].average_prec
        score = self.compute_score(ap=ap, min_val=min_val, max_val=max_val)
        return_str = self.return_string(
            ap=ap, score=score, max_score=15, min_val=min_val, max_val=max_val
        )
        return score, return_str

    @staticmethod
    def return_ec_string(ap, score, min_val):
        return f"AP = {ap :0.3f}: {score :>7d} (min {min_val :<4.2f})"

    @Case(score=3, extra_credit=True)
    def test_iou_ap0(self, min_val=0.5):
        """Average precision (iou > 0.5  c=0) [extra credit]"""
        ap = self.pr_iou[0].average_prec
        score = True if ap >= min_val else False
        return (
            score,
            self.return_ec_string(ap=ap, score=score, min_val=min_val),
        )

    @Case(score=3, extra_credit=True)
    def test_iou_ap1(self, min_val=0.3):
        """Average precision (iou > 0.5  c=1) [extra credit]"""
        ap = self.pr_iou[1].average_prec
        score = True if ap >= min_val else False
        return (
            score,
            self.return_ec_string(ap=ap, score=score, min_val=min_val),
        )

    @Case(score=3, extra_credit=True)
    def test_iou_ap2(self, min_val=0.6):
        """Average precision (iou > 0.5  c=2) [extra credit]"""
        ap = self.pr_iou[2].average_prec
        score = True if ap >= min_val else False
        return (
            score,
            self.return_ec_string(ap=ap, score=score, min_val=min_val),
        )
