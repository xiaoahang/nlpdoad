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
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
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

