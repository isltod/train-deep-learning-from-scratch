import sys

sys.path.append("b3-framework")

import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph


x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = "x"
y.name = "y"
y.backward(create_graph=True)

iters = 5

for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
gx.name = "gx" + str(iters + 1)
# gx에서 출발해서 creator, input, output 관계를 따라가며 그래프 만들기
plot_dot_graph(gx, verbose=False, to_file="tanh.png")
