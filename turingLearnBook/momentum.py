#  Copyright (c)
#  projectName: nlpdoad        # filename : momentum.py
#  Author: weihangzhang         # email: hannah.zz@qq.com
#  createDate : 2019/5/8 下午8:45
#  lastModified: 2019/5/8 下午8:45
#  desc: momentum是动量的意思，

import numpy as np


class Momentum:
    def __init__(self, lr=0.1, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

            for key in params.keys():
                self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
                params[key] += self.v[key]

# 实例变量v回哦保存物体的速度
# 初始化时，v什么都不保存，但点那个第一次调用update时，v会以字典变量的形式保存于参数结构相同的数据
