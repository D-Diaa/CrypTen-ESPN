#!/usr/bin/env python3
import numpy as np
import scipy
import torch


# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class AverageMeter:
    """Measures average of a value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.values = []

    def add(self, value, n=1):
        self.sum += value * n
        self.count += n
        self.values += [value] * n

    def mean_confidence_interval(self, confidence=0.95):
        a = 1.0 * np.array(self.values)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m - h, m + h

    def value(self):
        return self.sum / self.count


class AccuracyMeter:
    """Measures top-k accuracy of multi-class predictions."""

    def __init__(self, topk=(1,)):
        self.reset()
        self.topk = topk
        self.maxk = max(self.topk)

    def reset(self):
        self.values = []

    def add(self, output, ground_truth):
        # compute predicted classes (ordered):
        _, prediction = output.topk(self.maxk, 1, True, True)
        prediction = prediction.t()

        # store correctness values:
        correct = prediction.eq(ground_truth.view(1, -1).expand_as(prediction))
        self.values.append(correct[: self.maxk])

    def value(self):
        result = {}
        correct = torch.stack(self.values, 0)
        for k in self.topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            result[k] = correct_k.mul_(100.0 / correct.size(0))
        return result
