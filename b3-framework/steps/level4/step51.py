import sys

sys.path.append("b3-framework")

from dezero import Variable, as_variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
from dezero.datasets import Spiral, MNIST
from dezero import DataLoader
import numpy as np
import dezero

train_set = MNIST(train=True, transform=None)
test_set = MNIST(train=False, transform=None)
print(len(train_set), len(test_set))
