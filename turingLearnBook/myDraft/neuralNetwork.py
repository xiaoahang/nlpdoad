# 神经网络
# 输入层 输出层 隐藏层
# y=h(b+w1*x1+w2*x2)
# 激活函数 activation function ： 将输入信号的总和转换为输出信号（加权输入信号和偏置）
# 阶跃函数：激活函数以阈值为界，一旦输入超过阈值，就切换输入
# 感知机中，使用了阶跃函数作为激活函数

import numpy as np
import matplotlib.pyplot as plt


# 阶跃函数的图形
def step_function(x):
    return np.array(x > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)  # 在-5.0到5.0的范围内，以0.1为单位，生成numpy数组
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.title("step_function")
plt.show()


# sigmoid function
# h(x) = 1/1+exp(-x)    e是Napier常数2.7182
# sigmoid函数是神经网络常用函数之一
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


s = np.array([-1.0, 1.0, 2.0])
t = np.array([1.0, 2.0, 3.0])
print(1.0 + t)
print(1.0 / t)
print(sigmoid(s))

p = np.arange(-5.0, 5.0, 0.1)
q = sigmoid(p)
plt.plot(p, q)
plt.ylim(-0.1, 1.1)
plt.title("sigmoid")
plt.show()


# 阶跃函数和sigmoid函数都是非线性函数
# 激活函数必须使用非线性函数

# ReLU函数 rectified linear unit
# ReLU在输入大雨0时，直接输出该值；在输入小于0时，输出0
def relu(x):
    return np.maximum(0, x)


r = np.arange(-5.0, 5.0, 0.1)
u = relu(r)
plt.plot(r, u)
plt.title("ReLU")
plt.show()

# 多维数组
A = np.array([1, 2, 3, 4])
print(A)
np.ndim(A)  # 数组的维数
print(A.shape)
print(A.shape[0])
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)
print(B.shape[0])

# 矩阵乘法
C = np.array([[1, 2], [3, 4]])
D = np.array([[5, 6], [7, 8]])
E = np.dot(C, D)  # np.dot()接受NumPy数组作为参数，并返回数组的乘积
print(E)

# 2*3 和 3*2 的矩阵相乘, 结果不同
# 并且左侧数组的列数要和右侧数组的行数一致
# 拓展这个概念左侧数组的第一维要和右侧数组的第0维元素个数一致
# 3*2 dot 2*4 = 3*4
# 3*2 dot 2 = 3
# 2*3 dot 2*2 error
F = np.array([[1, 2], [3, 4], [5, 6]])
G = np.array([[1, 2, 3], [4, 5, 6]])
H = np.dot(F, G)
I = np.dot(G, F)
print(H)
print(I)

# 2*3 dot 2*2 error
try:
    J = np.dot(C, F)
except ValueError as e:
    print('error :', e)
finally:
    print('END')
# Traceback (most recent call last):
#   File "<input>", line 2, in <module>
# ValueError: shapes (2,2) and (3,2) not aligned: 2 (dim 1) != 3 (dim 0)
K = np.dot(F, C)
print(K)
# 3*2 dot 2*4 = 3*4
L = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
M = np.dot(F, L)
print(M)

# 神经网络的内积
# X   W  =  Y
# 2  2*3    3
_x = np.array([1, 2])
_w = np.array([[1, 3, 5], [2, 4, 6]])
_y = np.dot(_x, _w)
print(_y)  # [ 5 11 17]

# 3层神经网络的实现
# 神经网络的运算可以作为矩阵运算打包进行
# 第一层
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(W1.shape)
print(X.shape)
print(B1.shape)
print(A1)
print(Z1)
# 第二层
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print(Z1.shape)
print(W2.shape)
print(B2.shape)
print(A2)
print(Z2)


# 恒等函数 作为输出层的激活函数
# 一般滴，回归问题用恒等函数，二元分类问题可以使用sigmoid函数，多元分类问题用softmax函数
def identify_function(x):
    return x


W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identify_function(A3)


#
#
# 按照惯例整理下3层神经网络的实现
#
#
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identify_function(a3)
    return y


network = init_network()
x = np.array([1.5, 0.5])
y = forward(network, x)
print('y', y)


# 输出层的设计
# 分类问题中的softmax函数 yk = exp(ak) / sum(exp(ai)  (i:1~n)
# softmax 函数的实现
def softmax(a):
    c = np.max(a)  # 溢出对策
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


a = np.array([1010, 1000, 990])
y = softmax(a)
print(y)  # softmax函数输出的值是0.0到1.0之间的实数
print(np.sum(y))  # 并且softmax函数的输出值的总和是1
# 这样我们才把softmax函数的输出解释为概率

# 实现神经网络的推理处理，也叫神经网络的前向传播 forward propagation
# 手写数字的识别
# MNIST数据集


