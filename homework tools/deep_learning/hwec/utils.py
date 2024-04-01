import re
import string
from typing import Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader

# this adds a space and period to the vocabulary
vocab = string.ascii_lowercase + ' .'


def one_hot_faster(s: str):
    """
    Converts a string into a one-hot encoding

    :param s: a string with characters in vocab (all other characters will be ignored!)
    :type s: str
    :return: a once hot encoding Tensor r (len(vocab), len(s)), with r[j, i] = (s[i] == vocab[j])
    :rtype: torch.Tensor
    """
    import numpy as np
    # Not needed.  It's cleaned up and lowercase already
    # regex to remove non-vocab characters
    # s = re.sub(f'[^{vocab}]', '', s.lower())
    assert isinstance(s, str), f"Expected string, got {type(s)}"

    if len(s) == 0:
        return torch.zeros((len(vocab), 0))

    oht = torch.zeros(len(vocab), len(s))

    # can remove the loop to check each character since it comes in cleaned up
    for i, c in enumerate(s):
        oht[vocab.index(c), i] = 1.0

    return oht


def one_hot(s: Union[str, torch.Tensor]) -> torch.Tensor:
    """
    Converts a string into a one-hot encoding
    :param s: a string with characters in vocab (all other characters will be ignored!)
    :return: a once hot encoding Tensor r (len(vocab), len(s)), with r[j, i] = (s[i] == vocab[j])
    """
    if isinstance(s, str):
        import numpy as np
        if len(s) == 0:
            return torch.zeros((len(vocab), 0))

        return torch.as_tensor(
            np.array(list(s.lower()))[None, :] == np.array(list(vocab))[:, None]
        ).float()

    else:
        if len(s) == 0:
            return torch.zeros(len(vocab), 0)  # , device=s.device)

        # Assuming `vocab` is a string of vocabulary characters and `s` is a tensor of indices
        # Create a tensor of the same dtype as `s`, containing indices corresponding to `vocab`
        vocab_indices = torch.tensor([vocab.index(c) for c in vocab], device=s.device, dtype=s.dtype)

        # One-hot encode by comparing indices in `s` with `vocab_indices`
        one_hot_encoded = torch.zeros(len(vocab), len(s)).float()  # Pre-allocate a float tensor on the GPU
        for i, vocab_idx in enumerate(vocab_indices):
            one_hot_encoded[i] = (s == vocab_idx).float()  # Fill in the one-hot encoded tensor

        return one_hot_encoded


class SpeechDataset(Dataset):
    """
    Creates a dataset of strings from a text file.
    All strings will be of length max_len and padded with '.' if needed.

    By default this dataset will return a string, this string is not directly readable by pytorch.
    Use transform (e.g. one_hot) to convert the string into a Tensor.
    """

    def __init__(self, dataset_path, transform=None, max_len=250):
        with open(dataset_path, 'r', encoding='utf-8') as file:
            st = file.read()
        st = st.lower()
        reg = re.compile('[^%s]' % vocab)
        period = re.compile(r'[ .]*\.[ .]*')
        space = re.compile(r' +')
        sentence = re.compile(r'[^.]*\.')
        self.data = space.sub(' ', period.sub('.', reg.sub('', st)))
        if max_len is None:
            self.range = [(m.start(), m.end()) for m in sentence.finditer(self.data)]
        else:
            self.range = [(m.start(), m.start()+max_len) for m in sentence.finditer(self.data)]
            self.data += self.data[:max_len]
        if transform is not None:
            self.data = transform(self.data)

    def __len__(self):
        return len(self.range)

    def __getitem__(self, idx):
        s, e = self.range[idx]
        if isinstance(self.data, str):
            return self.data[s:e]
        return self.data[:, s:e]


def load_tcn_data(
    dataset_path: str,
    transform=one_hot,
    num_workers: int = 0,
    batch_size: int = 32,
    max_len=250,
    shuffle: bool = True,
    seed: Optional[int] = None,
    sample_rate: Optional[float] = None,
    **kwargs,
) -> DataLoader:
    if seed is not None:
        torch.manual_seed(seed)

    return DataLoader(
        SpeechDataset(dataset_path=dataset_path, transform=transform, max_len=max_len),
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        **kwargs,
    )


if __name__ == "__main__":
    data = SpeechDataset('data/train.txt', max_len=None)
    print('Dataset size ', len(data))
    for i in range(min(len(data), 10)):
        print(data[i])

    data = SpeechDataset('data/train.txt', transform=one_hot, max_len=None)
    print('Dataset size ', len(data))
    for i in range(min(len(data), 3)):
        print(data[i])
