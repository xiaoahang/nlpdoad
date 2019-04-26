# 感知机
# 简单的实现
import numpy as np


def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


print(AND(0, 0),
      AND(1, 0),
      AND(0, 1),
      AND(1, 1))

# 导入权重和偏置

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
print(w * x)
print(np.sum(w * x))
print(np.sum(w * x) + b)


# 使用权重和偏置的实现
def AND2(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# 非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# 或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# 到这里我们已经知道，使用感知机可以实现与门，或门，与非门 三种逻辑电路
# 接下来我们来考虑一下异或门
# 感知机的局限性在于它只能表示一条直线分割的空间
# 但是我们可以通过多层感知机来实现

# 异或门
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


print(XOR(0, 0), XOR(0, 1), XOR(1, 0), XOR(1, 1))

# 异或门是一种层结构的神经网络
