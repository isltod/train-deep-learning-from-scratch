import sys

sys.path.append("b3-framework")

from dezero import Variable, Model
import dezero.functions as F
import dezero.layers as L
import numpy as np
import matplotlib.pyplot as plt

model = L.Layer()
# 여기 괄호는 __init__()으로 전달...
model.l1 = L.Linear(5)
model.l2 = L.Linear(3)


def predict(model, x):
    # 여기 괄호는 __call__()로 전달...
    y = model.l1(x)
    y = F.sigmoid(y)
    y = model.l2(y)
    return y


# 이렇게 하면 들어있는 레이어 안에 있는 매개변수까지 다 반환
for p in model.params():
    print(p)

# cleargrads도 포함된 모든 레이어들을 다 돌면서 그 안의 매개변수들을 cleargrad
model.cleargrads()

# 다시 sin 회귀 문제를 새로 만든 Model 클래스를 이용해서 다루기...
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

lr = 0.2
iters = 10000
hidden_size = 10


class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)

    def forward(self, x):
        y = self.l1(x)
        y = F.sigmoid(y)
        y = self.l2(y)
        return y


model = TwoLayerNet(hidden_size, 1)

for i in range(iters):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    for p in model.params():
        p.data -= lr * p.grad.data

    if i % 1000 == 0:
        print(loss)

plt.scatter(x, y, s=10)
plt.xlabel("x")
plt.ylabel("y")
t = np.arange(0.0, 1.0, 0.01)[:, np.newaxis]
y_pred = model(t)
plt.plot(t, y_pred.data, color="r")
plt.tight_layout()
plt.show()
