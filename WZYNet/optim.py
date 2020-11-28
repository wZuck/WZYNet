# FileName: dataloader.py (proj1)
# CreateTime: 2020.11.20
# Author: © 王子翊 MF20330086@NJU

import numpy as np


class optimizer(object):
    def __init__(self, params):
        super(optimizer, self).__init__()
        self.params = params

    def step(self):
        raise NotImplemented

    def zero_grad(self):
        raise NotImplemented


class SGD(optimizer):
    ''' PyTorch Version SGD.    
    \\begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
    \end{aligned}
    The implementation of SGD with Momentum/Nesterov subtly differs from Sutskever et. al. and implementations in some other frameworks.
    '''

    def __init__(self, params, lr, momentum=0, weight_decay=0):
        super(SGD, self).__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        for param in params:
            param.momentum = self.momentum

    def step(self):
        # isinstance(object, classinfo)
        for param in self.params:
            param.velocity = param.momentum*param.velocity+param.gradient
            param.data = param.data - self.lr*param.velocity

    def zero_grad(self):
        for param in self.params:
            param.gradient = 0


class Adam(optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-9, weight_decay=0):
        super(Adam, self).__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        for param in params:  # Refer to https://arxiv.org/abs/1412.6980
            param.momentum = 0
            param.velocity = 0

    def step(self):
        for param in self.params:
            param.momentum = self.beta1*param.momentum + \
                (1-self.beta1)*param.gradient
            param.velocity = self.beta2*param.velocity + \
                (1-self.beta2)*(param.gradient**2)
            hat_momentum = param.momentum/(1-self.beta1)
            hat_velocity = param.velocity/(1-self.beta2)
            param.data = param.data - self.lr * \
                hat_momentum/(np.sqrt(hat_velocity)+self.eps)

    def zero_grad(self):
        for param in self.params:
            param.gradient = 0
