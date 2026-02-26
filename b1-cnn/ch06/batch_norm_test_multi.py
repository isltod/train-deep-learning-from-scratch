import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 여기선 데이터를 줄인다? 시간이 엄청 걸리나?
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
    # optimizer = SGD(lr=learning_rate)
    optimizer = Adam(lr=learning_rate)
    train_acc_list = []
    bn_train_acc_list = []

    # 1000/100 = 10 -> 100개씩 배치로 10번 돌리면 1000개 다 소진하므로 에포크란 개념...
    # 근데 실제론 랜덤 선택이므로 다 쓰는건 아닌데...
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0

    for i in range(1_000_000_000):
        # 1000개 중 100개 뽑아 [723, 539, 68, ...]
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        for _network in (bn_network, network):
            # 네트워크의 그라디언트는...각 노드별 미분치(그라디언트) 구하기
            grads = _network.gradient(x_batch, t_batch)
            # 그걸로 가중치 갱신
            optimizer.update(_network.params, grads)

        # 10번, 에포크 끝날 때마다 정확도 보고하고, 나갈지 결정
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
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
            if epoch_cnt >= max_epochs:
                break
    return train_acc_list, bn_train_acc_list


weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

plt.figure(figsize=(8, 8))

for i, w in enumerate(weight_scale_list):
    print("============== " + str(i + 1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)

    plt.subplot(4, 4, i + 1)
    plt.title("W:" + str(w), fontsize=6.5)
    plt.plot(x, bn_train_acc_list, label="Batch Normalization", markevery=2)
    plt.plot(
        x,
        train_acc_list,
        linestyle="--",
        label="Normal(without BatchNorm)",
        markevery=2,
    )
    plt.ylim(0, 1.0)

    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy", fontsize=6.5)

    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs", fontsize=6.5)

    plt.legend(loc="lower right", fontsize=6.5)

plt.subplots_adjust(wspace=0.2, hspace=0.3)
plt.show()
