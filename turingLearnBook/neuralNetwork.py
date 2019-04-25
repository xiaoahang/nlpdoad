# 神经网络
# 输入层 输出层 隐藏层
# y=h(b+w1*x1+w2*x2)
# 激活函数 activation function ： 将输入信号的总和转换为输出信号（加权输入信号和偏置）
# 阶跃函数：激活函数以阈值为界，一旦输入超过阈值，就切换输入
# 感知机中，使用了阶跃函数作为激活函数


# sigmoid function
# h(x) = 1/1+exp(-x)    e是Napier常数2.7182

# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0
# 以上代码只能接受浮点数，不能接受数组
import numpy as np
import matplotlib.pyplot as plt


# def step_function(x):
#     y = x > 0
#     return y.astype(np.int)


# x = np.array([-1.0, 1.0, 2.0])
# print(x)
# y = x > 0
# print(y)
# y = y.astype(np.int)
# print(y)


# 阶跃函数的图形
def step_function(x):
    return np.array(x > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)  # 在-5.0到5.0的范围内，以0.1为单位，生成numpy数组
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


# sigmoid函数是神经网络常用函数之一
# sigmoid函数的实现
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


s = np.array([-1.0, 1.0, 2.0])
print(sigmoid(s))
t = np.array([1.0, 2.0, 3.0])
print(1.0 + t)
print(1.0 / t)

p = np.arange(-5.0, 5.0, 0.1)
q = sigmoid(p)
plt.plot(p, q)
plt.ylim(-0.1, 1.1)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# 阶跃函数和sigmoid函数都是非线性函数
# 激活函数必须使用非线性函数
