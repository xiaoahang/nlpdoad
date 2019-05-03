#  Copyright (c)
#  projectName: nlpdoad        # filename : mul_layer.py
#  Author: weihangzhang         # email: hannah.zz@qq.com
#  createDate : 5/3/19 8:12 PM
#  lastModified: 5/3/19 8:12 PM
#  desc: 乘法层的实现


class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


apple = 100
apple_num = 2
tax = 1.1
# layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)

dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_tax_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)
