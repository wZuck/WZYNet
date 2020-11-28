# FileName: dataloader.py (proj1)
# CreateTime: 2020.11.20
# Author: © 王子翊 MF20330086@NJU

import unittest
import numpy as np
from WZYNet import show_plot


class TestUtils(unittest.TestCase):

    def test_show_plot(self):
        x = np.arange(-5, 5, 0.1)
        y = np.arange(-5, 5, 0.1)
        x, y = np.meshgrid(x, y)
        z = np.sin(x)-np.cos(y)
        show_plot(x, y, z)


if __name__ == "__main__":
    unittest.main()
