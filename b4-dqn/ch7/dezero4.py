import numpy as np
import dezero.layers as L
import dezero.functions as F
from dezero import Model
from dezero import optimizers
from matplotlib import pyplot as plt


linear = L.Linear(10)

batch_size, input_size = 100, 5
x = np.random.randn(batch_size, input_size)
y = linear(x)

print("y shape:", y.shape)

for param in linear.params():
    print(param.name, param.shape)

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y


model = TwoLayerNet(10, 1)
optimizer = optimizers.SGD(lr).setup(model)


for i in range(iters):
    y_pred = model.forward(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()

    if i % 1000 == 0:
        print(loss)

plt.scatter(x, y, s=10)
plt.xlabel("x")
plt.ylabel("y")
t = np.arange(0.0, 1.0, 0.01)[:, np.newaxis]
y_pred = model.forward(t)
plt.plot(t, y_pred.data, color="r")
plt.show()
