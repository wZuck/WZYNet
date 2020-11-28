# FileName: dataloader.py (proj1)
# CreateTime: 2020.11.20
# Author: © 王子翊 MF20330086@NJU

import numpy as np
from numpy import linalg as LA
import unittest


class Loss(object):
    def __init__(self):
        super(Loss, self).__init__()

    def __call__(self, *args):
        raise NotImplementedError(
            "Loss {} not implemented".format(self.__class__.__name__))


class NormLoss(Loss):
    def __init__(self, norm=2, reduction='mean'):
        super(NormLoss, self).__init__()
        self.norm = norm
        self.reduction = reduction

    def __call__(self, sources, targets):
        """
        Calculate N-Norm Loss.

        Args:
            sources: [N, *] source data.
            targets: [N, *] target data.
        """
        assert len(sources) == len(
            targets), 'Incompatible shape between sources and targets.'
        delta = sources-targets
        loss = LA.norm(delta, ord=self.norm, axis=1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # self.reduction == None
            return loss


class L1Loss(NormLoss):
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__(norm=1, reduction=reduction)
        self.norm = 1
        self.reduction = reduction

    def __call__(self, sources, targets):
        return super(L1Loss, self).__call__(sources, targets), np.sign(sources-targets)


class MSELoss(NormLoss):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__(norm=2, reduction=reduction)
        self.norm = 1
        self.reduction = reduction

    def __call__(self, sources, targets):
        return 0.5*super(MSELoss, self).__call__(sources, targets), sources-targets
