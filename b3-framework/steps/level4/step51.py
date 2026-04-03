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
import matplotlib.pyplot as plt


train_set = MNIST(train=True, transform=None)
test_set = MNIST(train=False, transform=None)
print(len(train_set), len(test_set))

x, t = train_set[0]
print(x.shape, type(x))
print(t)

plt.imshow(x.reshape(28, 28), cmap="gray")
plt.axis("off")
# plt.show()

max_epoch = 5
batch_size = 100
hidden_size = 1000

train_set = MNIST(train=True)
test_set = MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# x, t = train_set[0]
# print(x.shape, type(x))
# print(t)

# plt.imshow(x.reshape(28, 28), cmap="gray")
# plt.axis("off")
# plt.show()

# model = MLP((hidden_size, 10))
# optimizer = optimizers.SGD().setup(model)
model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print("epoch : {}".format(epoch + 1))
    print(
        "train loss : {:.4f}, accuracy : {:.4f}".format(
            sum_loss / len(train_set), sum_acc / len(train_set)
        )
    )

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print(
        "test loss : {:.4f}, accuracy : {:.4f}".format(
            sum_loss / len(test_set), sum_acc / len(test_set)
        )
    )
