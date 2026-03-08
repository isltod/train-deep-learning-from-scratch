import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 여기선 데이터를 줄인다?
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01


def __train(weight_init_std):
    # 배치 정규화 있는 네트워크와 없는 보통 네트워크 만들기...
    bn_network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_init_std,
        use_batchnorm=True,
    )
    network = MultiLayerNetExtend(
        input_size=784,
        hidden_size_list=[100, 100, 100, 100, 100],
        output_size=10,
        weight_init_std=weight_init_std,
    )
    optimizer = SGD(lr=learning_rate)
    # optimizer = Adam(lr=learning_rate)
    train_acc_list = []
    bn_train_acc_list = []

    # 에포크 당 반복 - 여기선 1000/100 = 10번
    # - 한 번에 100개씩 10번 돌면 훈련 데이터 다 쓴거고, 그게 한 에포크...
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    # 10억번 반복 - 근데 밑에서 max_epoch = 20 넘으면 중지시키니까, 실제론 10x20=200번 반복...
    for i in range(1_000_000_000):
        # 마스크는 1000개 중 100개 뽑는 걸로...shape는 (100,) -> [745, 213, ...]
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        # iter_per_epoch = 10이므로 10번마다 정확도 보고하고, 나갈지 결정
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            bn_train_acc_list.append(bn_train_acc)

            print(
                "epoch:"
                + str(epoch_cnt)
                + " | "
                + str(train_acc)
                + " - "
                + str(bn_train_acc)
            )
            epoch_cnt += 1
            if epoch_cnt > max_epochs:
                break
    return train_acc_list, bn_train_acc_list


# 로그스케일로 벡터 만들기 - 10^0 ~ 10^-4, 16개 숫자로 나누기
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

train_acc_list, bn_train_acc_list = __train(weight_scale_list[4])

plt.title("Training Accuracy")
plt.plot(x, bn_train_acc_list, label="Batch Normalization", markevery=2)
plt.plot(
    x, train_acc_list, label="Normal (without BatchNorm)", linestyle="--", markevery=2
)
plt.ylim(0, 1.0)
plt.xlim(0, max_epochs)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")

plt.show()
