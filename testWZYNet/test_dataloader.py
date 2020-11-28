# FileName: dataloader.py (proj1)
# CreateTime: 2020.11.20
# Author: © 王子翊 MF20330086@NJU

import unittest
from WZYNet import proj1_loader


class TestLoader(unittest.TestCase):

    def test_loader_iter(self):
        loader = proj1_loader()
        xs, y = loader.__iter__().__next__()
        print(xs, y)
        print(xs.shape, y.shape)


if __name__ == "__main__":
    unittest.main()
