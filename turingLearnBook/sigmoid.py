#  Copyright (c)
#  projectName: nlpdoad        # filename : sigmoid.py
#  Author: weihangzhang         # email: hannah.zz@qq.com
#  createDate : 5/4/19 12:17 PM
#  lastModified: 5/4/19 12:17 PM
#  desc: sigmoid 层
import numpy as np


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx