import sys

sys.path.append("b3-framework")

from dezero import Variable, as_variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
import numpy as np
import matplotlib.pyplot as plt

# x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = F.get_item(x, 1)
# print(y)

# y.backward()
# print(x.grad)

# indices = [0, 0, 1]
# y = F.get_item(x, indices)
# print(y)

# x.cleargrad()
# y.backward()
# print(x.grad)

# y = x[1]
# print(y)

# y = x[:, 2]
# print(y)

model = MLP((10, 3))
# x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
# y = model(x)
# print(y)


# def softmax1d(x):
#     x = as_variable(x)
#     y = F.exp(x)
#     sum_y = F.sum(y)
#     return y / sum_y


# x = np.array([0.2, -0.4])
# y = model(x)
# p = softmax1d(y)
# print(y)
# print(p)


# def softmax_simple(x, axis=1):
#     x = as_variable(x)
#     y = F.exp(x)
#     sum_y = F.sum(y, axis=axis, keepdims=True)
#     return y / sum_y


# x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
# y = model(x)
# p = softmax_simple(y)
# print(y)
# print(p)

x = np.array([[0.2, -0.4], [0.3, 0.5], [1.3, -3.2], [2.1, 0.3]])
t = np.array([2, 0, 1, 0])
# 이건 학습을 안하니 model에서 만든 가중치가 랜덤해서 책과는 다른 결과가 나온다...
y = model(x)
loss = F.softmax_cross_entropy_simple(y, t)
print(loss)
