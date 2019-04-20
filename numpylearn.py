# encoding: utf-8
import numpy as np

# x = np.array([1.0,2.0,3.0])
# print(x)
# type(x)

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x / 2)
# x 和 y 的元素个数要相同，不然会报错

# Numpy的N维数组
A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)  # 矩阵的形状
print(A.dtype)  # 矩阵元素的数据类型

B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)
print(A * 10)
# Numpy数据可以生成N维数组
# 数学上将一维数组称为向量，二维数组称为矩阵
# 张量 tensor

# 广播
# 形状不同的数组之间也可以进行运算
C = np.array([[1, 2], [3, 4]])
D = np.array([10, 20])
C * D

# 访问元素
# 元素的索引从0开始
E = np.array([[51, 55], [14, 19], [0, 4]])
print('E')
print(E)
print(E[0])
print(E[0][1])
for row in E:
    print('row ：', row)
E = E.flatten()  # 将E转化为一维数组
print(E)
print(E[np.array([0, 2, 4])])  # 获取索引为0，2，4的元素
print(E > 15)  # [ True  True False  True False False]
print(E[E > 15])  # [51 55 19]
