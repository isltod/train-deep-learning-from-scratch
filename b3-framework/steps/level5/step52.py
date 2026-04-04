import sys

sys.path.append("b3-framework")

from dezero import Variable, as_variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
from dezero.datasets import Spiral, MNIST
from dezero import DataLoader
import dezero
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import time

x = cp.arange(6).reshape(2, 3)
print(x)
y = x.sum(axis=1)
print(y)

# 넘파이 -> 쿠파이
n = np.array([1, 2, 3])
c = cp.asarray(n)
print(type(c) == cp.ndarray)

# 쿠파이 -> 넘파이
c = cp.array([1, 2, 3])
n = cp.asnumpy(c)
print(type(n) == np.ndarray)

# x가 넘파이 배열인 경우
x = np.array([1, 2, 3])
xp = cp.get_array_module(x)
print(xp == np)

# x가 쿠파이 배열인 경우
x = cp.array([1, 2, 3])
xp = cp.get_array_module(x)
print(xp == cp)

# GPU로 MNIST
max_epoch = 5
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

# GPU mode
if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    start = time.time()
    sum_loss = 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)

    elapsed_time = time.time() - start
    print(
        "epoch: {}, loss: {:.4f}, time: {:.4f}[sec]".format(
            epoch + 1, sum_loss / len(train_set), elapsed_time
        )
    )
