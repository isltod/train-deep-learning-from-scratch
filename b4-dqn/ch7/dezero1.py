import numpy as np
from dezero import Variable
import dezero.functions as F

x_np = np.array(5.0)
x = Variable(x_np)

y = 3 * x**2
print(y)

y.backward()
print(x.grad)

# 벡터 내적
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a, b = Variable(a), Variable(b)
c = F.matmul(a, b)
print(c)

# 행렬곱
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c = F.matmul(a, b)
print(c)
