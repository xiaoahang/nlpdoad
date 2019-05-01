# 学习算法的实现
# 步骤1 mini_batch
# 从训练数据中随机选出一部分数据，这部分数据称为mini-batch。我们的目标是减小mini-batch的损失函数的值

# 步骤2 计算梯度
# 为了减小mini-batch的损失函数的值，需要求出各个权重参数的梯度。梯度表示损失函数的值减小最多的方向。

# 步骤3 更新参数
# 将权重参数沿梯度方向进行微小更新。

# 步骤4 重复。 重复步骤1，2，3
# 以上方法又称随机梯度下降法 stochastic gradient descent （SGD）


# 神经网络的类
import sys, os

sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.1):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # 第一层权重
        self.params['b1'] = np.zeros(hidden_size)  # 第一层偏置
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)  # 第二层权重
        self.params['b2'] = np.zeros(output_size)  # 第二层偏置

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x ： 输入数据， t ：监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, y, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x ： 输入数据， t ：监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])  # 第一层权重梯度
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])  # 第一层偏置梯度
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])  # 第二层权重梯度
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])  # 第二层偏置梯度

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
# 784 = 28 * 28
# print(net.params['W1'].shape,
#       net.params['b1'].shape,
#       net.params['W2'].shape,
#       net.params['b2'].shape)

x = np.random.rand(100, 784)
y = net.predict(x)
