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

x = np.ones(5)
print(x)

y = F.dropout(x)
print(y)

with test_mode():
    y = F.dropout(x)
    print(y)
