# FileName: dataloader.py (proj1)
# CreateTime: 2020.11.20
# Author: © 王子翊 MF20330086@NJU

class Parameter(object):
    def __init__(self, data, required_grad=True):
        super(Parameter, self).__init__()
        self.data = data
        self.required_grad = required_grad
        self.velocity = 0
        self.gradient = 0
        self.momentum = 0


if __name__ == "__main__":
    pass
