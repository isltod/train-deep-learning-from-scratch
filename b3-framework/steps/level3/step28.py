import sys

sys.path.append("b3-framework")

import math
import numpy as np
from dezero import Function, Variable
from dezero.utils import plot_dot_graph


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0**2) ** 2 + (1 - x0) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

lr = 0.001
# 실제로는 5만번 정도 돌려야 정답인 1, 1에 가깝게 근접한다...
iters = 1000

for i in range(iters):
    # 변수 x0, x1 자체야 안 변하지만, 지금 문제에서는 경사하강법으로 그 데이터가 계속 변하므로...
    # 그걸 표시하고, 그 지점에서의 함수도 다시 만들어야 새 미분값을 구할 수 있다...
    print(x0, x1)
    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad

# plot_dot_graph(y, verbose=False, to_file="rosenbrock.png")
