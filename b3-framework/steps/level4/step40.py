import sys

sys.path.append("b3-framework")

import numpy as np
from dezero import Variable
import dezero.functions as F

from dezero.utils import sum_to

# aa = np.array([[[[[[1]]]]]])
# print(aa)
# print(aa.shape)
# bb = aa.squeeze((0, 1, 2))
# cc = sum_to(aa, (1, 2))
# print(bb)
# print(bb.shape)


# x = np.array([[1, 2, 3], [4, 5, 6]])
# y = sum_to(x, (1, 2))
# print(y)

x0 = Variable(np.array([1, 2, 3]))
x0 = Variable(np.random.randn(1, 2, 1, 1, 5))
x1 = Variable(np.array([[10], [20]]))
y = x0 + x1
print(y)

y.backward()
print(x1.grad)
print(x0.grad)
