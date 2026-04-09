import sys

sys.path.append("b3-framework")

from dezero import Variable, as_variable, Parameter
from dezero import optimizers
import dezero.functions as F
from dezero.layers import Layer
from dezero.models import MLP
from dezero.datasets import Spiral, MNIST
from dezero import DataLoader
import dezero
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import time
from dezero import test_mode

x1 = np.random.rand(1, 3, 7, 7)
col1 = F.im2col(x1, kernel_size=5, stride=1, pad=0, to_matrix=True)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7)
kernel_size = (5, 5)
stride = (1, 1)
pad = (0, 0)
col2 = F.im2col(x2, kernel_size, stride, pad, to_matrix=True)
print(col2.shape)

N, C, H, W = 1, 5, 15, 15
OC, KH, KW = 8, 3, 3

x = Variable(np.random.randn(N, C, H, W))
W = np.random.randn(OC, C, KH, KW)
y = F.conv2d_simple(x, W, b=None, stride=1, pad=1)
y.backward()
print(y.shape)
print(x.grad.shape)
