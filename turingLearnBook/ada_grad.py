#  Copyright (c)
#  projectName: nlpdoad        # filename : ada_grad.py
#  Author: weihangzhang         # email: hannah.zz@qq.com
#  createDate : 2019/5/8 下午9:06
#  lastModified: 2019/5/8 下午9:06
#  desc: 学习率衰减 learn rate decay
#  adaptive 会为每个参数适当地调整学习率

import numpy as np


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.lr is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

            for key in params.keys():
                self.h[key] += grads[key] * grads[key]
                params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
