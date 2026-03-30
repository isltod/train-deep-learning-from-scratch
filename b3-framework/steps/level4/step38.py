import sys

sys.path.append("b3-framework")

import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)

x = Variable(np.random.randn(1, 2, 3))
y = x.reshape((2, 3))
print(x)
print(x.shape)
print(y.shape)

y = x.reshape(2, 3)
print(y.shape)

y = x.reshape(3, 2)
print(y.shape)

y = x.reshape(6)
print(y.shape)

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.transpose(x)
print(y)
y.backward()
print(x.grad)

x = Variable(np.random.randn(2, 3))
y = x.transpose()
print(x)
print(y)
print(x.shape)
print(y.shape)

y = x.T
print(y)
print(x.shape)
print(y.shape)

A, B, C, D = 1, 2, 3, 4
x = Variable(np.random.randn(A, B, C, D))
y = x.transpose(2, 3, 1, 0)
print(x.shape)
print(y.shape)
y.backward(retain_grad=True)
print(x.grad.shape)
