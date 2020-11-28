# FileName: dataloader.py (proj1)
# CreateTime: 2020.11.20
# Author: © 王子翊 MF20330086@NJU

import numpy as np
import unittest


class proj1_loader(object):
    """
        [!Infinity Dataloader] Here i just implement a dataloader which only provides $x1,x2 \in (-5,5)$ and $y = sin(x1)-cos(x2)$.

        Args:
            batch_size (int, optional): Batch Size. Defaults to 32.
    """

    def __init__(self, batch_size=32, iterations=10000):
        super(proj1_loader, self).__init__()
        self.batch_size = batch_size
        self.iterations = iterations
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index >= self.iterations:
            self.index = 0
            raise StopIteration
        batch_x = []
        batch_y = []
        for _ in range(self.batch_size):
            x1 = np.random.random_sample(1)*10-5
            x2 = np.random.random_sample(1)*10-5
            xs = np.concatenate((x1, x2), axis=0)
            y = np.sin(x1)-np.cos(x2)
            batch_x.append(xs)
            batch_y.append(y)
        batch_x = np.vstack(batch_x)
        batch_y = np.vstack(batch_y)
        return batch_x, batch_y
