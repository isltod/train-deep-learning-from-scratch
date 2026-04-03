import sys

sys.path.append("b3-framework")

from dezero import Variable, as_variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
from dezero.datasets import Spiral
from dezero import DataLoader
import numpy as np
import dezero


class MyIterator:
    def __init__(self, max_cnt):
        self.max_cnt = max_cnt
        self.cnt = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt == self.max_cnt:
            raise StopIteration()
        self.cnt += 1
        return self.cnt


obj = MyIterator(5)
for x in obj:
    print(x)

batch_size = 10
max_epoch = 1

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

for epoch in range(max_epoch):
    for x, t in train_loader:
        print(x.shape, t.shape)
        break
    for x, t in test_loader:
        print(x.shape, t.shape)
        break

x = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
t = np.array([1, 2, 0])
acc = F.accuracy(x, t)
print(acc)

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    # 먼저 훈련 데이터...
    for x, t in train_loader:
        y = model(x)
        # 훈련 데이터의 손실과 정확도...
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

    # 다시 테스트 데이터인데...관련 변수 초기화하고...역전파 없이 계산...
    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            # 테스트 데이터의 손실과 정확도
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

        print(
            "test loss : {:.4f}, accuracy : {:.4f}".format(
                sum_loss / len(test_set), sum_acc / len(test_set)
            )
        )
