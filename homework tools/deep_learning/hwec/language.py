from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model
from . import utils

from heapq import heappush, heapreplace
import numpy as np
import torch
from torch.distributions import Categorical

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, *args, **kwargs):
        return x

device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_likelihood(model: LanguageModel, some_text: str):
    """
    Evaluate the log-likelihood of a given string.

    Hint: utils.one_hot might come in handy
    """
    raise NotImplementedError("log_likelihood")


def sample_random(model: LanguageModel, max_length: int = 100):
    """
    Your code here.

    Sample a random sentence from the language model.
    Terminate once you reach a period '.'
    """
    raise NotImplementedError("sample_random")


class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1) [1]
    h.add(2) [1,2]
    h.add(3) [2,3]
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def sort(self, reverse=True):
        self.elements = sorted(self.elements, reverse=reverse)

    def add(self, e):
        if any(e[1] == b[1] for b in self.elements):
            return
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif e > self.elements[0]:
            heapreplace(self.elements, e)


def beam_search(
    model: LanguageModel,
    beam_size: int,
    n_results: int = 10,
    max_length: int = 30,
    average_log_likelihood: bool = False
):
    """
    Use beam search for find the highest likelihood generations, such that:
      * No two returned sentences are the same
      * the `log_likelihood` of each returned sentence is as large as possible
    """
    raise NotImplementedError("beam_search")


if __name__ == "__main__":
    """
      Some test code.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    # for s in ['abcdef', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.']:
    for s in ['yes', 'we', 'can', 'abcdef', 'abcgdee', 'bcdefg']:  # , '.abcdef', 'fedcba.']:
        print(s, float(log_likelihood(lm, s)))
    print()

    for i in range(10):
        s = sample_random(lm)
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100):
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100, average_log_likelihood=True):
        print(s, float(log_likelihood(lm, s)) / len(s))
