#  Copyright (c)
#  projectName: nlpdoad        # filename : relu.py
#  Author: weihangzhang         # email: hannah.zz@qq.com
#  createDate : 5/3/19 9:24 PM
#  lastModified: 5/3/19 9:24 PM
#  desc: ReLU实现


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x < 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


# 正向传播的输入值小于0，则反向传播的的值为0
# 正向传播大于0

