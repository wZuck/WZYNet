# FileName: dataloader.py (proj1)
# CreateTime: 2020.11.20
# Author: © 王子翊 MF20330086@NJU

from WZYNet import parameter
import unittest
from WZYNet import Linear, ModuleSequence, Parameter
import numpy as np


class TestModules(unittest.TestCase):
    def test_Linear(self):
        x = np.array([[5, 7]])
        fc = Linear(2, 3)
        fc.w = Parameter(np.array([[1, 2, 3], [3, 2, 1]]))
        fc.b = Parameter(np.array([[10, 20, 30]]))
        y = fc(x)
        self.assertEqual(y.all(), np.array([[36, 44, 52]]).all())

    def test_ModuleSequence(self):
        x = np.array([[5, 7]])
        fc = Linear(2, 3)
        fc.w = Parameter(np.array([[1, 2, 3], [3, 2, 1]]))
        fc.b = Parameter(np.array([[10, 20, 30]]))
        model = ModuleSequence([fc])
        y = fc(x)
        self.assertEqual(y.all(), np.array([[36, 44, 52]]).all())
        # print(model.parameters())
        for p in model.parameters():
            print(type(p))


if __name__ == "__main__":
    unittest.main()
