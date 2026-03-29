import sys

sys.path.append("b3-framework")

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph


def goldstein(x, y):
    z = (
        1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)
    ) * (
        30
        + (2 * x - 3 * y) ** 2
        * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )
    return z


x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))

# 이게 그래프로 그릴 계산이고
z = goldstein(x0, x1)
z.backward()


# 이건 변수 이름
x0.name = "x0"
x1.name = "x1"
z.name = "z"

# 한 방에 dot 문자열 얻고 그림 만들기
plot_dot_graph(z, verbose=False, to_file="goldstein.png")
