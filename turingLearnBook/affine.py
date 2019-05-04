#  Copyright (c)
#  projectName: nlpdoad        # filename : affine.py
#  Author: weihangzhang         # email: hannah.zz@qq.com
#  createDate : 5/4/19 12:49 PM
#  lastModified: 5/4/19 12:49 PM
#  desc: Affine 层的实现
#  神经网络的正向传播中进行的矩阵乘积运算在几何学领域被称为 " 仿射变换 "
#  因此，这里将进行仿射变换的处理实现为 " Affine层"


import numpy as np


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx
