# FileName: dataloader.py (proj1)
# CreateTime: 2020.11.20
# Author: © 王子翊 MF20330086@NJU

import unittest
from WZYNet import L1Loss, MSELoss
import numpy as np


class TestLosses(unittest.TestCase):
    '''Just a little test case.
    '''

    def test_L1Loss(self):
        a = np.array([[1], [2]])
        b = np.array([[3], [4]])
        norm_loss1 = L1Loss(reduction='mean')
        self.assertEqual(norm_loss1(a, b), 2.0)

    def test_MSELoss(self):
        a = np.array([[1], [2]])
        b = np.array([[3], [4]])
        norm_loss2 = MSELoss(reduction='mean')
        self.assertEqual(norm_loss2(a, b), 1.0)


if __name__ == "__main__":
    unittest.main()
