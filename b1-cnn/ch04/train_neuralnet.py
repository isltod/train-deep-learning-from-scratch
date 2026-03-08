import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# 하이퍼파라미터
iters_num = 10000
train_size = x_train.shape[0]  # 60,000
batch_size = 100
learning_rate = 0.1

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in tqdm(range(iters_num)):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)  # [24, 59, 31...] 100개
    x_batch = x_train[batch_mask]  # (100, 784)
    t_batch = t_train[batch_mask]  # (100, 10)

    # 기울기 계산
    # 이 버전으로는 계산 시간이 너무 많이 걸려서 현실적으로 안된다...
    grad = network.numerical_gradient(x_batch, t_batch)
    # 일단 이 버전 내용 공부는 5장으로 넘기고, 여기서는 그냥 사용
    # grad = network.gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

# 그래프 그리기
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.ylim(0, 9)
plt.xlim(0, 10000)
plt.show()
