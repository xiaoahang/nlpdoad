# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt

# matplotplib 是用于绘制图形的库

# 生成数据
x = np.arange(0, 6, 0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()

y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label='sin')
plt.plot(x, y2, linestyle="--", label='cos')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("sin & cos")
plt.legend()
plt.show()


# pyplot中还提供了显示图像的方法imshow()
# 另外可以使用matplotlib.image模块中的imread()方法读入图像
from  matplotlib.image import imread
img = imread('lena.png')
plt.imshow(img)
plt.show()

