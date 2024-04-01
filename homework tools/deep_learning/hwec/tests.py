"""
EDIT THIS FILE AT YOUR OWN RISK!
It will not ship with your code, editing it will only change the test cases locally, and might make you fail our
remote tests.
"""
from os import path
import torch
import numpy as np
import string

from .grader import Grader, Case, MultiCase

vocab = string.ascii_lowercase+' .'


def one_hot(s: str):
    if len(s) == 0:
        return torch.zeros((len(vocab), 0))
    return torch.as_tensor(np.array(list(s.lower()))[None, :] == np.array(list(vocab))[:, None]).float()


class LanguageGrader(Grader):
    """Language modeling"""

    def __init__(self, *a, **ka):
        super().__init__(*a, **ka)
        self.bigram = self.module.Bigram()
        # self.tcn = self.module.TCN()
        self.tcn = torch.load(path.join(path.dirname(path.abspath(__file__)), 'tcn.th'))

        class Dummy(self.module.LanguageModel):
            # fake model that simply returns the vocabulary in order
            def predict_all(self, text):
                # return a tensor of shape (len(vocab), len(text)+1) set to 1e-5
                r = 1e-5*torch.ones(len(vocab), len(text)+1)
                # then for each char, set [index + 1, i + 1] to 1
                for i, s in enumerate(text):
                    r[(vocab.index(s)+1)%len(vocab), i+1] = 1
                # don't understand these settings
                # so with the letter 'a' as the text we will end up with
                # r[0, 0] = 2
                # r[0, 1] = 1e-5
                # r[1, 0] = 1
                # r[1, 1] = 1
                r[0, 0] = 2
                r[1, 0] = 1
                # print(r)
                return (r/r.sum(dim=0, keepdim=True)).log()

        self.dummy = Dummy()

    @Case(score=10)
    def test_log_likelihood(self):
        """log_likelihood"""
        ll = self.module.log_likelihood

        def check(m, s, r):
            l = ll(m, s)
            model_name = m.__class__.__name__
            # print(f"\t\t-- Checking {model_name} for '{s}' got {l :0.6f} expected {r}")
            assert abs(r-l) < 1e-2, \
                (
                    f"wrong log likelihood for {model_name}: {s} got {l :0.6f} expected {r}\n"
                    # f"{''.join([f'{c:0.6f} ' for c in m.predict_all(s).flatten()])}"
                )

        check(self.bigram, 'yes', -8.914730)
        check(self.bigram, 'we', -3.708824)
        check(self.bigram, 'can', -7.696493)
        # check(self.tcn, 'yes', -9.960992)  # 8.914730)
        # check(self.tcn, 'we', -6.668854)  # .708824)
        # check(self.tcn, 'can', -10.140140)  # .696493)
        check(self.dummy, 'abcdef', -0.406903)
        check(self.dummy, 'abcdee', -11.919827)
        check(self.dummy, 'bcdefg', -1.100051)

        def check_sum(m, length):
            # this increase exponentially with length, so we only check for small lengths
            # even 4 elements can take a very long time with 28^4 possibilities
            all_str = []
            all_sub_str = ['']
            while len(all_sub_str):
                s = all_sub_str.pop()
                if len(s) == length:
                    all_str.append(s)
                else:
                    for c in vocab:
                        all_sub_str.append(s + c)

            # print(all_str)
            exp = [np.exp(ll(m, s).detach().cpu().numpy()) for s in all_str]
            # print(exp)
            l = np.sum(exp)
            print(f"\t\t-- Checking {m.__class__.__name__} for all strings of length {length} got {l}")
            assert abs(1-l) < 1e-2, "Log likelihood for '%s' does not sum to 1" % (m.__class__.__name__)

        check_sum(self.bigram, length=0)
        check_sum(self.tcn, length=0)
        check_sum(self.dummy, length=0)
        check_sum(self.bigram, length=1)
        check_sum(self.tcn, length=2)
        check_sum(self.dummy, length=1)
        check_sum(self.bigram, length=2)
        check_sum(self.tcn, length=2)
        check_sum(self.dummy, length=2)

    @Case(score=10)
    def test_sample_random(self):
        """sample_random"""
        ll = self.module.log_likelihood
        sample = self.module.sample_random

        def check(m, min_likelihood):
            samples = [sample(m) for i in range(10)]
            sample_ll = np.median([float(ll(m, s))/len(s) for s in samples])
            # print(samples, sample_ll, min_likelihood)

            model_name = m.__class__.__name__
            print(f"\t\t-- Checking {model_name} for random samples got {sample_ll} expected > {min_likelihood}")
            assert sample_ll > min_likelihood, "'%s' : min ll %f got %f"%(model_name, min_likelihood, sample_ll)

        print("\t-- Probability test")
        check(self.bigram, -2.5)
        check(self.tcn, -3.5)
        check(self.dummy, -0.05)

        print("\t-- Sampling test")
        samples = [sample(self.dummy) for i in range(100)]
        print("\t-- Checking for bias")
        chars = [s[10] if len(s) > 10 else 'a' for s in samples]
        # There is a 1 in 100 billion chance this fill fail
        assert any([abs(sum([c == 'k' for c in chars[i: i+10]])-6.666) < 2 for i in range(0, 100, 10)]), \
            "Your samples seem biased"
        # There is a 1 in 100 billion chance this fill fail
        assert any([abs(sum([c == 'l' for c in chars[i: i+10]])-3.333) < 2 for i in range(0, 100, 10)]), \
            "Your samples seem biased"

    @Case(score=20)
    def test_beam_search(self):
        """beam_search"""
        ll = self.module.log_likelihood
        bs = self.module.beam_search

        def check(m, n, min_log_likelihood, average_log_likelihood=False):
            samples = bs(m, 100, n, max_length=30, average_log_likelihood=average_log_likelihood)
            model_name = m.__class__.__name__
            print(model_name)
            assert len(samples) == n, (
                f"{len(samples)} samples expected {n}: "
                f"Model {model_name} ALL {average_log_likelihood}"
            )
            assert all([s not in samples[:i] for i, s in enumerate(samples)]), 'Beam search returned duplicates'
            med_ll = np.median([float(ll(m, s))*(1./len(s) if average_log_likelihood else 1.) for s in samples])
            print(
                f"\t-- Beam search for {model_name} with n={n} samples"
                f" and ALL={average_log_likelihood}: {med_ll}"
                f"\n\t\t{samples}"
            )
            assert med_ll > min_log_likelihood, "Beam search failed to find high likelihood samples"

        # check(self.bigram, 10, -7.5, False)
        # check(self.bigram, 10, -1.5, True)
        # ToDo: put me back
        check(self.tcn, 10, -10., False)
        check(self.tcn, 10, -10., True)
        # check(self.dummy, 10, -12., False)
        # check(self.dummy, 10, -0.4, True)
        # check(self.dummy, 2, -0.8, False)
        # check(self.dummy, 2, -0.1, True)


class TCNGrader(Grader):
    """TCN"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("Loading model")
        self.tcn = self.module.TCN()
        self.tcn.eval()

    @MultiCase(score=3, b=[1, 16, 128], length=[0, 10, 20])
    def test_forward(self, b, length):
        """TCN.forward"""
        one_hot = (torch.randint(len(vocab), (b, 1, length)) == torch.arange(len(vocab))[None, :, None]).float()
        output = self.tcn(one_hot)
        assert output.shape == (b, len(vocab), length+1), \
            'TCN.forward output shape expected (%d, %d, %d) got %s'%(b, len(vocab), length+1, output.shape)

    @MultiCase(score=3, s=['', 'a', 'ab', 'abc'])
    def test_predict_all(self, s):
        """TCN.predict_all"""
        output = self.tcn.predict_all(s)
        output = output.to('cpu')

        assert output.shape == (len(vocab), len(s)+1), \
            f"{s}: shape expected ({len(vocab)},{len(s)+1}) got {output.shape}"
        assert np.allclose(output.exp().sum(dim=0).detach(), 1), "log likelihoods do not sum to 1"

    @MultiCase(score=7, s=['united'])  # , 'states', 'yes', 'we', 'can'])
    def test_consistency(self, s):
        """TCN.predict_next/TCN.predict_all consistency"""
        all_ll = self.tcn.predict_all(s).detach().cpu().numpy()
        for i in range(len(s)+1):
            seq = s[:i]
            ll = self.tcn.predict_next(seq).detach().cpu().numpy()
            assert np.allclose(all_ll[:, i], ll), \
                f"predict_next != predict_all for {s}:'{seq}'\n{list(zip(ll, all_ll[:, i]))}"

    # @MultiCase(score=7, i=range(1))  # 00))
    # def test_causal(self, i):
    #     """TCN.forward causality"""
    #     input = torch.zeros(len(vocab), 100)
    #     input[:, i] = float('NaN')
    #     output = self.tcn(input[None])[0]
    #     is_nan = (output != output).any(dim=0)
    #     assert not is_nan[:i+1].any(), "Model is not causal, information leaked forward in time"
    #     assert is_nan[i+3:].any(), "Model does not consider a temporal extend > 2"

    @MultiCase(score=2, i=range(5, 95))
    def test_shape(self, i):
        """TCN.forward shape"""
        input = torch.zeros(len(vocab), i)
        output = self.tcn(input[None])[0]
        assert (output.shape[0] == input.shape[0]) and (output.shape[1] == input.shape[1]+1), \
            f"Got {output.shape} for input shape {input.shape}: expected [{input.shape[0]}, {input.shape[1] + 1}]!"

    @MultiCase(score=5, i=range(10,90))
    def test_causal(self, i):
        """TCN.forward causality"""
        input = torch.zeros(len(vocab), 100)
        input[:, i] = float('NaN')
        output = self.tcn(input[None])[0]
        is_nan = (output != output).any(dim=0)
        assert not is_nan[:i+1].any(), "Model is not causal, information leaked forward in time"
        assert is_nan[i+3:].any(), "Model does not consider a temporal extend > 2"


class TrainedTCNGrader(Grader):
    """TrainedTCN"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tcn = self.module.load_model()
        self.tcn.eval()
        self.data = self.module.SpeechDataset('data/valid.txt')

    @Case(score=40)
    def test_nll(self):
        """Accuracy"""
        lls = []
        for s in self.data:
            ll = self.tcn.predict_all(s)
            lls.append(float((ll[:, :-1]*one_hot(s)).sum()/len(s)))
        nll = -np.mean(lls)
        return max(2.3-max(nll, 1.3), 0), 'nll = %0.3f' % nll
