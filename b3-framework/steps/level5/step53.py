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

x = np.array([1, 2, 3])
np.save("test.npy", x)
x = np.load("test.npy")
print(x)

x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])
np.savez("test.npz", x1=x1, x2=x2)

arrays = np.load("test.npz")
x1 = arrays["x1"]
x2 = arrays["x2"]
print(x1, x2)

data = {"x1": x1, "x2": x2}
# 딕셔너리를 전개해서 전달할 때, 즉 위처럼 전달할 때 ** 사용한다고...
np.savez("test.npz", **data)

arrays = np.load("test.npz")
x1 = arrays["x1"]
x2 = arrays["x2"]
print(x1, x2)

layer = Layer()
l1 = Layer()
l1.p1 = Parameter(np.array([1]))
layer.l1 = l1
layer.p2 = Parameter(np.array([2]))
layer.p3 = Parameter(np.array([3]))

params_dict = {}
layer._flatten_params(params_dict)
print(params_dict)

# 가중치 저장 실습
max_epoch = 3
batch_size = 100

train_set = MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)
model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

# 가중치 읽기 - 없으면 그냥 나오고...
model.load_weights("my_mlp.npz")

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

# 가중치 저장
model.save_weights("my_mlp.npz")
