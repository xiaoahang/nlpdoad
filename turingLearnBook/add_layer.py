#  Copyright (c)
#  projectName: nlpdoad        # filename : add_layer.py
#  Author: weihangzhang         # email: hannah.zz@qq.com
#  createDate : 5/3/19 9:15 PM
#  lastModified: 5/3/19 9:15 PM
#  desc: 加法层的实现


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
