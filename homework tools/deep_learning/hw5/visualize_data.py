from .utils import SuperTuxDataset

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main(args):
    dataset = SuperTuxDataset(args.dataset)

    # idxes = np.random.randint(len(dataset), size=args.N)
    idxes = np.array([0, 1, 2, 3, 4])
    f, axes = plt.subplots(1, args.N, figsize=(4 * args.N, 4))

    for i, idx in enumerate(idxes):
        img, point = dataset[idx]
        print(point)
        WH2 = np.array([img.size(-1), img.size(-2)]) / 2
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].axis("off")
        circle = Circle(WH2 * (point + 1), ec="r", fill=False, lw=2)
        axes[i].add_patch(circle)

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("dataset")
    parser.add_argument("-N", type=int, default=5)

    main(parser.parse_args())
