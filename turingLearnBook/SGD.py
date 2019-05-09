#  Copyright (c)
#  projectName: nlpdoad        # filename : SGD.py
#  Author: weihangzhang         # email: hannah.zz@qq.com
#  createDate : 2019/5/8 下午8:35
#  lastModified: 2019/5/8 下午8:35
#  desc: stochastic gradient descent 随机梯度下降法


#
class SGD:
    """lr 指learning rate"""

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


