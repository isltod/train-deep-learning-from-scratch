import sys

sys.path.append("b3-framework")

from dezero import Variable, as_variable, Parameter
from dezero import optimizers
import dezero.functions as F
from dezero.layers import Linear, RNN
from dezero import DataLoader
import dezero
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import time
from dezero import test_mode
from dezero import Model

rnn = RNN(10)
x = np.random.rand(1, 1)
h = rnn(x)
print(h.shape)


class SimpleRNN(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.rnn = RNN(hidden_size)
        self.fc = Linear(out_size)

    def reset_state(self):
        self.rnn.reset_state()

    def forward(self, x):
        h = self.rnn(x)
        y = self.fc(h)
        return y


seq_data = [np.random.randn(1, 1) for _ in range(1000)]
# 입력과 정답
xs = seq_data[0:-1]
ts = seq_data[1:]

model = SimpleRNN(10, 1)

loss, cnt = 0, 0
for x, t in zip(xs, ts):
    y = model(x)
    loss += F.mean_squared_error(y, t)
    cnt += 1
    if cnt == 2:
        model.cleargrads()
        loss.backward()
        break

train_set = dezero.datasets.SinCurve(train=True)
print(len(train_set))
print(train_set[0])
print(train_set[1])
print(train_set[2])

# xs = [example[0] for example in train_set]
# ts = [example[1] for example in train_set]
# plt.plot(np.arange(len(xs)), xs, label="xs")
# plt.plot(np.arange(len(ts)), ts, label="ts")
# plt.show()

max_epoch = 100
hidden_size = 100
bptt_length = 30

seqlen = len(train_set)
model = SimpleRNN(hidden_size, 1)
optimizer = optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    model.reset_state()
    loss, count = 0, 0

    for x, t in train_set:
        x = x.reshape(1, 1)
        y = model(x)
        loss += F.mean_squared_error(y, t)
        count += 1

        if count % bptt_length == 0 or count == seqlen:
            model.cleargrads()
            loss.backward()
            loss.unchain_backward()
            optimizer.update()

    avg_loss = float(loss.data) / count
    print("| epoch %d | loss %f" % (epoch + 1, avg_loss))

# 정작 모델은 sin 커브로 학습했는데, 예측을 cos 커브로...그래도 맞는건 어차피 그놈이 그놈이라?
xs = np.cos(np.linspace(0, 4 * np.pi, 1000))
model.reset_state()
pred_list = []

with dezero.no_grad():
    for x in xs:
        x = np.array(x).reshape(1, 1)
        y = model(x)
        pred_list.append(float(y.data.item()))

plt.plot(np.arange(len(xs)), xs, label="y=cos(x)")
plt.plot(np.arange(len(xs)), pred_list, label="predict")
plt.show()
#
