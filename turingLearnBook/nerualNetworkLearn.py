import numpy as np
import sys, os

sys.path.append(os.pardir)
from dataset.mnist import load_mnist


# 损失函数 loss function 是表示神经网络性能的'恶劣程度'的指标
# 最有名的损失函数 均方误差 mean squared error

# 均方误差的实现
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 设 2 为正确解
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
print(np.ndim(y))
print(y)
y = y.reshape(1, y.size)
print(y)
M = mean_squared_error(np.array(y), np.array(t))
print(M)


# 交叉熵误差 cross entropy error
def cross_entropy_error(y, t):
    delta = 1e-7  # .0000001
    return -np.sum(t * np.log(y + delta))


# mini-batch
# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
# train_size = x_train.shape[0]
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_size)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]
#
# print(np.random.choice(6000, 10))


# mini-batch版交叉熵误差的实现
def batch_cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


#
def batch_cross_entropy_error_normalize(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# 导数的实现
# 不好的实现 数值微分

def numerical_diff(f, x):
    h = 10e-50
    return (f(x + h) - f(x - h)) / (2 * h)


# 梯度的实现
def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # 生成和x形状相同的数组

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值
    return grad


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


n1 = numerical_gradient(function_2, np.array([3.0, 4.0]))
n2 = numerical_gradient(function_2, np.array([0.0, 2.0]))
n3 = numerical_gradient(function_2, np.array([3.0, 0.0]))
print(n1, n2, n3)


# 梯度下降法
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


init_x = np.array([-3.0, 4.0])
G = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
print(G)
