#  Copyright (c)
#  projectName: nlpdoad        # filename : softmax_with_loss.py
#  Author: weihangzhang         # email: hannah.zz@qq.com
#  createDate : 2019/5/6 下午11:23
#  lastModified: 2019/5/6 下午11:23
#  desc:


# 前面我们提到过，softmax函数会将输入值进行正规化后再输出
# 神经网络中进行的处理有推理（inference）和学习两个阶段。神经网络的推理一般不使用softmax层。
# softmax层，交叉熵误差cross entropy error作为损失函数

from common.functions import softmax, cross_entropy_error


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx
