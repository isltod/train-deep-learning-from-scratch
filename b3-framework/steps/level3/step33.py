import sys

sys.path.append("b3-framework")

import numpy as np
from dezero import Function, Variable


def f(x):
    y = x**4 - 2 * x**2
    return y


x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    # 여기부터 2차 미분 계산하는 새 계산이긴 한데...
    gx = x.grad

    # 이걸 새 계산이라고 위로 올리면 gx가 None을 참조해 예외 발생..
    x.cleargrad()

    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data
