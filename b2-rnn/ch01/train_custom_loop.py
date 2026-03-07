import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from common.np import *
from common.config import GPU
from common.optimizer import SGD
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt
from dataset import spiral

# 하이퍼파라미터
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# 데이터 로드
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

# 학습 관련 변수들
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

# 학습 시작
for epoch in range(max_epoch):
    # 데이터 섞기
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters * batch_size : (iters + 1) * batch_size]
        batch_t = t[iters * batch_size : (iters + 1) * batch_size]

        # 기울기 구해서
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        if (iters + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print(
                "| epoch %d | iter %d / %d | loss %.2f"
                % (epoch + 1, iters + 1, max_iters, avg_loss)
            )
            loss_list.append(avg_loss)
            # loss_list.append(avg_loss.get())
            total_loss, loss_count = 0, 0

h = 0.001
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
y_min, y_max = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
# cupy 사용하면 여기서 또 cp.asarray(numpy_array) 필요하고...
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)

# cupy 모드였으면 matplotlib를 위해서 cpu 버전으로 변경...
if GPU:
    import numpy as np

    x = x.get()
    for i in range(len(loss_list)):
        loss_list[i] = loss_list[i].get()
    xx = xx.get()
    yy = yy.get()
    Z = Z.get()


plt.plot(np.arange(len(loss_list)), loss_list, label="train")
plt.xlabel("iterations (x10)")
plt.ylabel("loss")
plt.show()

plt.contourf(xx, yy, Z)
plt.axis("off")

x, t = spiral.load_data()
if GPU:
    x = x.get()
    t = t.get()
N = 100
CLS_NUM = 3
markers = ["o", "x", "^"]
for i in range(CLS_NUM):
    plt.scatter(
        x[i * N : (i + 1) * N, 0],
        x[i * N : (i + 1) * N, 1],
        s=40,
        marker=markers[i],
    )
plt.show()
