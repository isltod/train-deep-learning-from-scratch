import sys

sys.path.append("b3-framework")

from dezero import Variable, as_variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
import dezero.datasets
import numpy as np
import matplotlib.pyplot as plt
import math


max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

x, t = dezero.datasets.get_spiral(train=True)
print(x.shape)
print(t.shape)
print(x[10], t[10])
print(x[110], t[110])

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(x)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size : (i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        # loss가 평균이니까 배치 곱하면 전체 loss
        sum_loss += float(loss.data) * len(batch_t)

    # 누적 loss를 다시 데이터 크기로 나눠서 평균이라...뭔가 이상하네...
    avg_loss = sum_loss / data_size
    print("epoch %d, loss %.2f" % (epoch + 1, avg_loss))

h = 0.001
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

with dezero.no_grad():
    score = model(X)
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

N, CLS_NUM = 100, 3
markers = ["o", "x", "^"]
colors = ["orange", "blue", "green"]
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i, 0], x[i, 1], s=40, marker=markers[c], color=colors[c])
plt.show()
